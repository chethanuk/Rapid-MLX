import Foundation
import Observation

/// Incremental Markdown document for the message currently streaming.
///
/// ## Why this exists
///
/// The transcript still mutates `messages[i].content` on every coalesced SSE
/// batch, but `ChatView` removes that growing field from `MessageRow`'s
/// presentation snapshot. This store is the only high-frequency body
/// dependency for the streamed prose.
///
/// Measured on a 5 760-character answer streamed at ~313 chars/second: the
/// stream reader's `await MainActor.run` waited **55.5 ms** per hop while the
/// closure it was waiting to run took **0.0 ms**. The main thread was not
/// doing our work; it was rebuilding view trees. End to end the reply took
/// **181 s** against an 18 s transmission. The same fixture in the prototype
/// this renderer came from took 19.9 s, and its hops waited 2.1 ms.
///
/// The live row reads stable compiled chunks and one mutable tail from this
/// store instead. Uneven transport batches first enter a presentation buffer;
/// a display link releases complete graphemes to the mutable tail on screen
/// frames. An SSE batch therefore no longer invalidates the row chrome or
/// recompiles the settled prefix.
///
/// Messages that originated in this store retain their compiled segments after
/// completion. Restored history still compiles once through
/// `TextKitMarkdownView` because it has no live document to preserve.
@MainActor
@Observable
final class StreamingMarkdownStore {

    /// Stable chunks plus the one mutable tail for the active message.
    private(set) var document = StreamingMarkdownDocument()
    /// Which message `document` belongs to. Nil when nothing is streaming.
    private(set) var messageID: UUID?
    /// The owner of `document` remains available after streaming finishes so
    /// the settled row can keep the same renderer until another response starts.
    /// Older rows return to the one-shot renderer instead of retaining a second
    /// compiled copy of every answer for the lifetime of `ChatView`.
    private(set) var documentMessageID: UUID?

    private var presentationBuffer = StreamingTextPresentationBuffer()
    private var isFinishing = false

    var segments: [StreamingMarkdownDocument.Segment] { document.segments }
    var result: MarkdownResult { document.result }

    func segments(for messageID: UUID) -> [StreamingMarkdownDocument.Segment] {
        documentMessageID == messageID ? document.segments : []
    }

    func hasDocument(for messageID: UUID) -> Bool {
        documentMessageID == messageID
    }

    var receivedText: String { presentationBuffer.receivedText }
    var pendingGraphemeCount: Int { presentationBuffer.pendingGraphemeCount }
    var isPresentationActive: Bool { presentationBuffer.hasPendingText }

    /// Note new streamed text. It becomes visible on display frames rather
    /// than at the transport's uneven delivery cadence.
    func enqueue(id: UUID, text: String) {
        if documentMessageID != id {
            // A new turn receives a fresh identity space and document. The
            // completed row has already settled; `ChatView` moves it back to
            // the one-shot renderer rather than retaining every live document.
            document = StreamingMarkdownDocument()
            documentMessageID = id
            presentationBuffer = StreamingTextPresentationBuffer()
        }
        messageID = id
        isFinishing = false
        if presentationBuffer.receive(text) == .reset {
            document = StreamingMarkdownDocument()
        }
    }

    /// Finalise the current tail without rebuilding the stable prefix.
    ///
    /// Keep the final document under the same message ID. `ChatView` mounts one
    /// `MessageRow` for streaming and completion, so committing the final tail
    /// updates that row in place instead of constructing a settled renderer.
    func finish(id: UUID? = nil, finalText: String? = nil) {
        if let id, documentMessageID != id { return }
        if let finalText {
            synchronizeReceivedText(finalText)
        }
        messageID = nil
        isFinishing = true
        finalizeIfDrained()
    }

    /// Reconcile a completed row with the authoritative message model.
    /// Completion-time transforms such as startup failures and grounding-source
    /// appendages do not travel through `streamingBody`, so they arrive here.
    func synchronizeCompleted(id: UUID, text: String) {
        guard documentMessageID == id else { return }
        if !document.isFinished || presentationBuffer.hasPendingText {
            finish(id: id, finalText: text)
            return
        }
        guard document.receivedSource != text else { return }

        document = Self.finishedDocument(text)
        presentationBuffer = Self.drainedPresentationBuffer(text)
        messageID = nil
        isFinishing = false
    }

    /// Drop the live document when its message leaves the visible conversation.
    func reset() {
        document = StreamingMarkdownDocument()
        messageID = nil
        documentMessageID = nil
        presentationBuffer = StreamingTextPresentationBuffer()
        isFinishing = false
    }

    /// Advance presentation by one real or test-controlled display frame.
    func advancePresentationFrame(duration: TimeInterval) {
        if let delta = presentationBuffer.presentFrame(
            duration: duration,
            isFinishing: isFinishing
        ) {
            var next = document
            next.append(delta)
            document = next
        }
        finalizeIfDrained()
    }

    private func finalizeIfDrained() {
        guard isFinishing, !presentationBuffer.hasPendingText else { return }
        var next = document
        next.finish()
        document = next
        isFinishing = false
    }

    /// Bring the presentation input up to the final model value. This check is
    /// intentionally paid only at completion: validating a growing prefix on
    /// every transport update would restore the quadratic scan this pipeline
    /// exists to avoid.
    private func synchronizeReceivedText(_ text: String) {
        guard presentationBuffer.receivedText != text else { return }
        if text.hasPrefix(presentationBuffer.receivedText) {
            if presentationBuffer.receive(text) == .reset {
                document = StreamingMarkdownDocument()
            }
            return
        }

        presentationBuffer = StreamingTextPresentationBuffer()
        _ = presentationBuffer.receive(text)
        document = StreamingMarkdownDocument()
    }

    private static func finishedDocument(_ text: String) -> StreamingMarkdownDocument {
        var document = StreamingMarkdownDocument()
        document.append(text)
        document.finish()
        return document
    }

    private static func drainedPresentationBuffer(
        _ text: String
    ) -> StreamingTextPresentationBuffer {
        var buffer = StreamingTextPresentationBuffer()
        _ = buffer.receive(text)
        _ = buffer.drainAll()
        return buffer
    }
}
