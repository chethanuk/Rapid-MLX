import Foundation

/// Incremental Markdown state for one streaming assistant response.
///
/// Only the final top-level Markdown node remains mutable. Once a following
/// sibling establishes a boundary, the preceding source is compiled into an
/// immutable chunk and is never parsed again. This bounds repeated work to the
/// current paragraph, list, table, or fenced block instead of the full reply.
struct StreamingMarkdownDocument: Sendable {

    struct StableBlock: Identifiable, Equatable, Sendable {
        let id: Int
        let source: String
        let result: MarkdownResult
    }

    /// One independently-rendered document chunk. The mutable tail keeps its
    /// ID when it is committed, so SwiftUI can preserve the TextKit view that
    /// was already showing it and mount only the newly-created tail.
    struct Segment: Identifiable, Equatable, Sendable {
        let id: Int
        let result: MarkdownResult
        let isMutable: Bool
    }

    struct CompilationStats: Equatable, Sendable {
        private(set) var stableFragmentCompilations = 0
        private(set) var mutableTailCompilations = 0
        private(set) var splitProbes = 0
        private(set) var totalCompiledUTF8Bytes = 0
        private(set) var largestCompiledFragmentUTF8Bytes = 0

        mutating func recordSplitProbe() {
            splitProbes += 1
        }

        mutating func recordStable(_ source: String) {
            stableFragmentCompilations += 1
            record(source)
        }

        mutating func recordTail(_ source: String) {
            mutableTailCompilations += 1
            record(source)
        }

        private mutating func record(_ source: String) {
            let count = source.utf8.count
            totalCompiledUTF8Bytes += count
            largestCompiledFragmentUTF8Bytes = max(largestCompiledFragmentUTF8Bytes, count)
        }
    }

    private let compiler: MarkdownCompiler
    private(set) var mutableID = 0
    private var revision = 0
    private var splitBoundaryTracker = SplitBoundaryTracker()

    private(set) var receivedSource = ""
    private(set) var stableBlocks: [StableBlock] = []
    private(set) var mutableSource = ""
    private(set) var mutableResult: MarkdownResult = .empty
    private(set) var compilationStats = CompilationStats()
    private(set) var isFinished = false

    init(compiler: MarkdownCompiler = MarkdownCompiler()) {
        self.compiler = compiler
    }

    /// The renderable snapshot. Stable chunks retain their identity separately;
    /// this flattened result exists for parity checks and the future UI bridge.
    var result: MarkdownResult {
        MarkdownResult(
            items: Self.coalescingAdjacentImages(
                stableBlocks.flatMap(\.result.items) + mutableResult.items
            ),
            revision: revision
        )
    }

    var segments: [Segment] {
        var output = stableBlocks.map {
            Segment(id: $0.id, result: $0.result, isMutable: false)
        }
        if !mutableResult.items.isEmpty {
            output.append(Segment(
                id: mutableID,
                result: mutableResult,
                isMutable: true
            ))
        }
        return output
    }

    mutating func append(_ delta: String) {
        guard !delta.isEmpty, !isFinished else { return }
        let shouldProbeForSplit = splitBoundaryTracker.consume(delta)
        receivedSource += delta
        mutableSource += delta

        revision += 1
        if shouldProbeForSplit {
            commitStablePrefixIfAvailable()
            splitBoundaryTracker.reset(to: mutableSource)
        }
        compileMutableTail(isComplete: false)
    }

    /// Commit the final tail in place. This never recompiles the stable prefix.
    mutating func finish() {
        guard !isFinished else { return }
        revision += 1
        if !mutableSource.isEmpty {
            let final = compile(mutableSource, isComplete: true, stable: true)
            if !final.items.isEmpty {
                stableBlocks.append(StableBlock(
                    id: mutableID,
                    source: mutableSource,
                    result: final
                ))
                mutableID += 1
            }
        }
        mutableSource = ""
        mutableResult = .empty
        isFinished = true
    }

    private mutating func commitStablePrefixIfAvailable() {
        while true {
            compilationStats.recordSplitProbe()
            guard let split = compiler.topLevelStreamingSplit(mutableSource) else { return }
            let stable = compile(split.stablePrefix, isComplete: true, stable: true)
            if !stable.items.isEmpty {
                stableBlocks.append(StableBlock(
                    id: mutableID,
                    source: split.stablePrefix,
                    result: stable
                ))
                mutableID += 1
            }
            mutableSource = split.mutableTail
        }
    }

    /// A stable split requires a blank line followed by the first character of
    /// a new block. Track that cheap lexical event so ordinary display frames
    /// do not parse the tail once for probing and again for rendering.
    private struct SplitBoundaryTracker: Sendable {
        private var trailingLineBreaks = 0

        mutating func consume(_ text: String) -> Bool {
            var establishedBoundary = false
            for character in text {
                if character.isNewline {
                    trailingLineBreaks += 1
                } else if character == " " || character == "\t" {
                    continue
                } else {
                    if trailingLineBreaks >= 2 { establishedBoundary = true }
                    trailingLineBreaks = 0
                }
            }
            return establishedBoundary
        }

        mutating func reset(to source: String) {
            self = SplitBoundaryTracker()
            _ = consume(source)
        }
    }

    private mutating func compileMutableTail(isComplete: Bool) {
        guard !mutableSource.isEmpty else {
            mutableResult = .empty
            return
        }
        mutableResult = compile(mutableSource, isComplete: isComplete, stable: false)
    }

    private mutating func compile(
        _ source: String, isComplete: Bool, stable: Bool
    ) -> MarkdownResult {
        if stable {
            compilationStats.recordStable(source)
        } else {
            compilationStats.recordTail(source)
        }
        return compiler.compile(source, revision: revision, isComplete: isComplete)
    }

    /// `MarkdownCompiler` merges adjacent image-only paragraphs. Stable chunks
    /// are compiled independently, so preserve that document-level behavior
    /// when flattening them back into one result.
    private static func coalescingAdjacentImages(_ items: [MarkdownItem]) -> [MarkdownItem] {
        var output: [MarkdownItem] = []
        for item in items {
            if case let .images(next) = item,
               case let .images(previous)? = output.last {
                output[output.count - 1] = .images(.init(
                    urls: previous.urls + next.urls,
                    altTexts: previous.altTexts + next.altTexts
                ))
            } else {
                output.append(item)
            }
        }
        return output
    }
}
