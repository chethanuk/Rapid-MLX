import Foundation
import Testing
@testable import Rapid

@Suite("Streaming Markdown store")
@MainActor
struct StreamingMarkdownStoreTests {

    @Test("Accumulated updates append only their suffix")
    func accumulatedTextBecomesDocumentDeltas() {
        let id = UUID()
        let store = StreamingMarkdownStore()

        store.enqueue(id: id, text: "First paragraph.\n\nSec")
        presentAll(store)
        let stable = store.document.stableBlocks[0]
        let stableCompilations = store.document.compilationStats.stableFragmentCompilations

        store.enqueue(id: id, text: "First paragraph.\n\nSecond paragraph grows.")
        presentAll(store)

        #expect(store.document.receivedSource == "First paragraph.\n\nSecond paragraph grows.")
        #expect(store.document.stableBlocks == [stable])
        #expect(
            store.document.compilationStats.stableFragmentCompilations
                == stableCompilations
        )
        #expect(store.document.mutableSource == "Second paragraph grows.")
    }

    @Test("Finishing preserves the final segments for row handoff")
    func finishKeepsFinalSnapshot() {
        let id = UUID()
        let store = StreamingMarkdownStore()
        let source = "Intro.\n\nFinal **answer**."

        store.enqueue(id: id, text: source)
        store.finish()
        presentAll(store)

        #expect(store.messageID == nil)
        #expect(store.documentMessageID == id)
        #expect(store.hasDocument(for: id))
        #expect(store.document.isFinished)
        #expect(store.document.mutableSource.isEmpty)
        #expect(store.segments.allSatisfy { !$0.isMutable })
        #expect(store.segments(for: id) == store.segments)
        #expect(store.result.items == MarkdownCompiler().compile(source).items)
    }

    @Test("Finishing reconciles transport backlog with the authoritative body")
    func finishUsesAuthoritativeBody() {
        let id = UUID()
        let store = StreamingMarkdownStore()
        let partial = "Answer body"
        let final = partial + "\n\nSources:\n- [Rapid](https://rapidmlx.ai)"

        store.enqueue(id: id, text: partial)
        store.finish(id: id, finalText: final)
        presentAll(store)

        #expect(store.document.receivedSource == final)
        #expect(store.result.items == MarkdownCompiler().compile(final).items)
    }

    @Test("A completed document accepts a later authoritative correction")
    func completedDocumentSynchronizesLaterContent() {
        let id = UUID()
        let store = StreamingMarkdownStore()

        store.enqueue(id: id, text: "Initial")
        store.finish()
        presentAll(store)
        store.synchronizeCompleted(id: id, text: "Startup failure explanation")

        #expect(store.document.receivedSource == "Startup failure explanation")
        #expect(store.document.isFinished)
    }

    @Test("A new message releases the prior completed document")
    func newMessageReleasesCompletedDocument() {
        let store = StreamingMarkdownStore()
        let firstID = UUID()
        let secondID = UUID()

        store.enqueue(id: firstID, text: "Old first.\n\nOld second.")
        store.finish()
        presentAll(store)
        store.enqueue(id: secondID, text: "New answer")
        presentAll(store)

        #expect(store.messageID == secondID)
        #expect(store.documentMessageID == secondID)
        #expect(store.document.receivedSource == "New answer")
        #expect(store.document.stableBlocks.isEmpty)
        #expect(store.segments.map(\.id) == [0])
        #expect(!store.hasDocument(for: firstID))
        #expect(store.segments(for: firstID).isEmpty)
    }

    @Test("A detectable non-monotonic update resets safely")
    func shorterUpdateResetsDocument() {
        let id = UUID()
        let store = StreamingMarkdownStore()

        store.enqueue(id: id, text: "Long first paragraph.\n\nLong second paragraph.")
        presentAll(store)
        store.enqueue(id: id, text: "Replacement")
        presentAll(store)

        #expect(store.document.receivedSource == "Replacement")
        #expect(store.document.stableBlocks.isEmpty)
        #expect(store.document.mutableSource == "Replacement")
    }

    @Test("Transport input stays buffered until a presentation frame")
    func inputWaitsForDisplayFrame() {
        let store = StreamingMarkdownStore()
        let id = UUID()

        store.enqueue(id: id, text: "Buffered text")

        #expect(store.receivedText == "Buffered text")
        #expect(store.document.receivedSource.isEmpty)
        #expect(store.pendingGraphemeCount == "Buffered text".count)

        store.advancePresentationFrame(duration: 1.0 / 60.0)

        #expect(!store.document.receivedSource.isEmpty)
        #expect("Buffered text".hasPrefix(store.document.receivedSource))
    }

    @Test("Finishing drains over frames before finalizing in place")
    func finishDrainsBeforeFinalizing() {
        let store = StreamingMarkdownStore()
        let id = UUID()
        let source = String(repeating: "stream ", count: 30)

        store.enqueue(id: id, text: source)
        store.finish()

        #expect(!store.document.isFinished)
        #expect(store.isPresentationActive)

        presentAll(store)

        #expect(store.document.isFinished)
        #expect(store.document.receivedSource == source)
        #expect(!store.isPresentationActive)
    }

    @Test("Reset releases the active document and presentation backlog")
    func resetReleasesDocument() {
        let id = UUID()
        let store = StreamingMarkdownStore()

        store.enqueue(id: id, text: "Buffered response")
        store.reset()

        #expect(store.documentMessageID == nil)
        #expect(store.messageID == nil)
        #expect(!store.hasDocument(for: id))
        #expect(!store.isPresentationActive)
    }

    private func presentAll(
        _ store: StreamingMarkdownStore,
        maxFrames: Int = 30
    ) {
        for _ in 0..<maxFrames where store.isPresentationActive {
            store.advancePresentationFrame(duration: 1.0 / 60.0)
        }
        #expect(!store.isPresentationActive)
    }
}
