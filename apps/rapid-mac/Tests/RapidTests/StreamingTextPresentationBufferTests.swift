import Foundation
import Testing
@testable import Rapid

@Suite("Streaming text presentation buffer")
struct StreamingTextPresentationBufferTests {
    private let frameDuration = 1.0 / 60.0

    @Test("A transport burst is spread across display frames")
    func burstIsPresentedAcrossFrames() {
        let source = String(repeating: "abcdefghij", count: 12)
        var buffer = StreamingTextPresentationBuffer()
        var deltas: [String] = []

        #expect(buffer.receive(source) == .appended)
        while let delta = buffer.presentFrame(duration: frameDuration) {
            deltas.append(delta)
        }

        #expect(deltas.count > 1)
        #expect(deltas.count <= 8)
        #expect(deltas.joined() == source)
        #expect(buffer.presentedText == source)
    }

    @Test("Presentation never splits extended grapheme clusters")
    func preservesGraphemeBoundaries() {
        let graphemes = ["👨‍👩‍👧‍👦", "é", "🇨🇳", "好"]
        let source = graphemes.joined()
        var buffer = StreamingTextPresentationBuffer(
            configuration: .init(targetLatency: 1, completionDrainDuration: 1)
        )
        var deltas: [String] = []

        buffer.receive(source)
        while let delta = buffer.presentFrame(duration: frameDuration) {
            deltas.append(delta)
        }

        #expect(deltas == graphemes)
        #expect(deltas.allSatisfy { $0.count == 1 })
    }

    @Test("Adaptive rate bounds a large backlog")
    func adaptiveRateBoundsBacklog() {
        var buffer = StreamingTextPresentationBuffer()
        let source = String(repeating: "字", count: 600)
        var frames = 0

        buffer.receive(source)
        while buffer.hasPendingText {
            _ = buffer.presentFrame(duration: frameDuration)
            frames += 1
        }

        #expect(frames <= 8)
        #expect(buffer.presentedText == source)
    }

    @Test("Completion drains the remaining text within its deadline")
    func completionDrainIsBounded() {
        var buffer = StreamingTextPresentationBuffer(
            configuration: .init(targetLatency: 1, completionDrainDuration: 0.15)
        )
        let source = String(repeating: "tail", count: 100)
        var frames = 0

        buffer.receive(source)
        while buffer.hasPendingText {
            _ = buffer.presentFrame(duration: frameDuration, isFinishing: true)
            frames += 1
        }

        #expect(frames <= 9)
        #expect(buffer.presentedText == source)
    }

    @Test("A non-monotonic update resets received and presented state")
    func replacementResetsState() {
        var buffer = StreamingTextPresentationBuffer()

        buffer.receive("Long accumulated response")
        _ = buffer.presentFrame(duration: frameDuration)
        #expect(buffer.receive("Replacement") == .reset)

        #expect(buffer.receivedText == "Replacement")
        #expect(buffer.presentedText.isEmpty)
        #expect(buffer.pendingText == "Replacement")
        #expect(buffer.pendingGraphemeCount == "Replacement".count)
    }
}
