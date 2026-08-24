import Foundation
import Testing
@testable import Rapid

@Suite("Streaming Markdown document")
struct StreamingMarkdownDocumentTests {

    @Test("A following top-level block commits the stable prefix")
    func commitsStablePrefix() {
        var document = StreamingMarkdownDocument()

        document.append("First paragraph.")
        let mutableID = document.segments.first?.id

        document.append("\n\nSecond")

        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].id == 0)
        #expect(document.stableBlocks[0].id == mutableID)
        #expect(document.stableBlocks[0].source.contains("First paragraph."))
        #expect(document.mutableSource == "Second")
        #expect(document.segments.map(\.id) == [0, 1])
        #expect(document.segments.map(\.isMutable) == [false, true])

        let stable = document.stableBlocks[0]
        let stableCompilations = document.compilationStats.stableFragmentCompilations
        document.append(" paragraph continues without closing.")

        #expect(document.stableBlocks == [stable])
        #expect(document.compilationStats.stableFragmentCompilations == stableCompilations)
        #expect(document.mutableSource == "Second paragraph continues without closing.")
    }

    @Test("Byte-sized streaming produces the same final document as one-shot compilation")
    func characterReplayMatchesOneShotCompile() {
        let source = """
        # Streaming title

        A paragraph with **bold**, _emphasis_, `code`, and [Rapid](https://rapidmlx.ai).

        - first item
        - second item

        ```swift
        let greeting = "hello"
        print(greeting)
        ```

        | model | size |
        |:------|-----:|
        | small | 1 GB |

        最后一段包含中文和 $x_1$。
        """
        var document = StreamingMarkdownDocument()

        for character in source {
            document.append(String(character))
        }
        document.finish()

        let expected = MarkdownCompiler().compile(source)
        #expect(document.result.items == expected.items)
        #expect(document.receivedSource == source)
        #expect(document.mutableSource.isEmpty)
        #expect(document.isFinished)
        #expect(Set(document.stableBlocks.map(\.id)).count == document.stableBlocks.count)
    }

    @Test("SSE-sized chunks preserve final Markdown parity")
    func chunkReplayMatchesOneShotCompile() {
        let chunks = [
            "Intro with an image.\n\n",
            "![one](https://example.com/one.png)\n\n",
            "![two](https://example.com/two.png)\n\n",
            "> quoted ",
            "text\n\n",
            "Final paragraph.",
        ]
        let source = chunks.joined()
        var document = StreamingMarkdownDocument()

        for chunk in chunks {
            document.append(chunk)
        }
        document.finish()

        #expect(document.result.items == MarkdownCompiler().compile(source).items)
    }

    @Test("A later reference definition can still style an earlier paragraph")
    func referenceDefinitionsKeepEarlierSourceMutable() {
        let source = """
        Read [the docs] before continuing.

        A second paragraph proves a block boundary.

        [the docs]: https://rapidmlx.ai/docs
        """
        var document = StreamingMarkdownDocument()

        for character in source {
            document.append(String(character))
        }
        document.finish()

        #expect(document.result.items == MarkdownCompiler().compile(source).items)
        #expect(document.stableBlocks.first?.source.contains("[the docs]") == true)
    }

    @Test("A complete inline link does not block stable prefix commitment")
    func inlineLinkCanCommit() {
        var document = StreamingMarkdownDocument()

        document.append("Read [the docs](https://rapidmlx.ai/docs).\n\nNext paragraph")

        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].source.contains("[the docs](https://rapidmlx.ai/docs)"))
        #expect(document.mutableSource == "Next paragraph")
    }

    @Test("Code brackets do not block stable prefix commitment")
    func codeBracketsCanCommit() {
        let source = """
        Intro paragraph.

        Use `arr[0]` for the first value.

        ```swift
        let pivot = values[indexes[0]]
        ```

        Next paragraph
        """
        var document = StreamingMarkdownDocument()

        document.append(source)

        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].source.contains("arr[0]"))
        #expect(document.stableBlocks[0].source.contains("indexes[0]"))
        #expect(document.mutableSource == "Next paragraph")
    }

    @Test("An unresolved reference holds only its own suffix mutable")
    func unresolvedReferenceCommitsSafeLeadingBlocks() {
        var document = StreamingMarkdownDocument()

        document.append("Safe paragraph.\n\nRead [**the docs**].\n\nFollowing paragraph")

        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].source.contains("Safe paragraph."))
        #expect(!document.stableBlocks[0].source.contains("the docs"))
        #expect(document.mutableSource.hasPrefix("Read [**the docs**]."))

        document.append(
            "\n\n[**the docs**]: https://rapidmlx.ai/docs\n\nAfter the definition"
        )

        #expect(document.stableBlocks.contains { $0.source.contains("[**the docs**]:") })
        #expect(document.mutableSource == "After the definition")
        document.finish()
        let source = document.receivedSource
        #expect(document.result.items == MarkdownCompiler().compile(source).items)
    }

    @Test("Ordinary character frames do not probe for a structural split")
    func proseFramesSkipSplitProbe() {
        var document = StreamingMarkdownDocument()

        for character in "A paragraph arriving one character at a time." {
            document.append(String(character))
        }
        #expect(document.compilationStats.splitProbes == 0)

        document.append("\n")
        document.append("\n")
        document.append("N")
        let probesAtBoundary = document.compilationStats.splitProbes
        #expect(probesAtBoundary == 2)

        for character in "ext paragraph keeps growing." {
            document.append(String(character))
        }
        #expect(document.compilationStats.splitProbes == probesAtBoundary)
    }

    @Test("Code-heavy bracket streams retain incremental work bounds")
    func codeBracketsRetainIncrementalBounds() {
        func replay(expression: String) -> StreamingMarkdownDocument {
            var document = StreamingMarkdownDocument()
            for index in 0..<40 {
                document.append(
                    "## Part \(index)\n\nUse `\(expression)` in this step.\n\n"
                )
            }
            return document
        }

        let bracketed = replay(expression: "arr[0]")
        let control = replay(expression: "arr.first")

        #expect(bracketed.stableBlocks.count == control.stableBlocks.count)
        #expect(bracketed.stableBlocks.count >= 30)
        #expect(
            bracketed.compilationStats.largestCompiledFragmentUTF8Bytes
                <= control.compilationStats.largestCompiledFragmentUTF8Bytes + 16
        )
        #expect(bracketed.compilationStats.splitProbes == control.compilationStats.splitProbes)
    }

    @Test("Unfinished inline syntax and fences remain in the mutable tail")
    func unfinishedSyntaxStaysMutable() {
        var document = StreamingMarkdownDocument()

        document.append("Committed paragraph.\n\n**unfinished")
        #expect(document.stableBlocks.count == 1)
        #expect(document.mutableSource == "**unfinished")
        let stableID = document.stableBlocks[0].id

        document.append(" bold** and `inline")
        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].id == stableID)
        #expect(document.mutableSource.hasSuffix("`inline"))

        document.append(" code`\n\n```swift")
        #expect(document.stableBlocks.count == 2)
        #expect(document.mutableSource == "```swift")
        #expect(document.mutableResult.items.isEmpty)

        document.append("\nlet value = 1")
        #expect(document.stableBlocks.count == 2)
        #expect(document.mutableResult.items.contains {
            if case .code = $0 { return true }
            return false
        })
    }

    @Test("Unicode CRLF source locations split on a valid String boundary")
    func unicodeCRLFSplit() {
        let source = "你好，世界。\r\n\r\nSecond paragraph with café."
        var document = StreamingMarkdownDocument()

        document.append(source)

        #expect(document.stableBlocks.count == 1)
        #expect(document.stableBlocks[0].source.contains("你好，世界。"))
        #expect(document.mutableSource == "Second paragraph with café.")
        document.finish()
        #expect(document.result.items == MarkdownCompiler().compile(source).items)
    }

    @Test("Finishing never recompiles an already committed prefix")
    func finishOnlyCompilesTail() {
        var document = StreamingMarkdownDocument()
        document.append("Stable.\n\nMutable tail")
        let stableBeforeFinish = document.stableBlocks
        let stableCompilations = document.compilationStats.stableFragmentCompilations

        document.finish()

        #expect(Array(document.stableBlocks.prefix(stableBeforeFinish.count)) == stableBeforeFinish)
        #expect(
            document.compilationStats.stableFragmentCompilations
                == stableCompilations + 1
        )
        #expect(document.stableBlocks.last?.source == "Mutable tail")
    }

    @Test("Committing a paragraph preserves its rendered height")
    @MainActor
    func committedParagraphKeepsItsLayout() {
        let options = MarkdownOptions.assistantTranscript()
        let width: CGFloat = 500
        var document = StreamingMarkdownDocument()
        document.append("First paragraph.")
        let mutable = document.segments[0]
        let mutableHeight = renderedTextHeight(
            for: mutable.result, options: options, width: width
        )

        document.append("\n\nSecond paragraph.")
        let committed = document.segments[0]
        let committedHeight = renderedTextHeight(
            for: committed.result, options: options, width: width
        )

        #expect(committed.id == mutable.id)
        #expect(committed.result.items == mutable.result.items)
        #expect(committedHeight == mutableHeight)
    }

    @MainActor
    private func renderedTextHeight(
        for result: MarkdownResult,
        options: MarkdownOptions,
        width: CGFloat
    ) -> CGFloat {
        let blocks = result.items.compactMap { item -> MarkdownItem.TextBlock? in
            guard case let .text(block) = item else { return nil }
            return block
        }
        let renderer = MarkdownTextRenderer(options: options)
        renderer.setBlocks(blocks)
        return renderer.measureHeight(width: width)
    }
}
