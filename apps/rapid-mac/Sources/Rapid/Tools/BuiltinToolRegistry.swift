import Foundation

/// Concrete ``ToolRegistry`` that ships the built-in tools the chat
/// surface exposes:
///
///   * ``weather`` — no approval, hits Open-Meteo over HTTPS
///   * ``web_search`` — no approval, backend per ``WebSearchConfig``
///     (Keenable keyless by default; Parallel / Tavily / Brave with a
///     key; DuckDuckGo backstop)
///   * ``browse`` — USER-approved per fetch (``BrowseApprovalStore``),
///     SSRF-guarded, byte-capped
///
/// One instance is constructed by ``RapidApp`` and shared by the chat
/// view model. Filesystem / shell tools are deliberately absent: this
/// build has no ``SandboxManager``, and a tool that touches the user's
/// disk must not ship without one.
@MainActor
final class BuiltinToolRegistry: ToolRegistry {
    typealias WebSearchRunner = (
        _ arguments: String,
        _ provider: WebSearchProvider,
        _ apiKey: String?
    ) async -> ToolCallResult

    /// Model-visible audit note for the one credential-recovery transition.
    /// It deliberately names the mode change without echoing any credential.
    static let rejectedKeyRecoveryNote = "Note: Rapid removed a rejected saved Keenable key and retried this search using Keenable's keyless mode. Future searches will stay keyless until a new key is saved."

    /// Per-invocation approval gate for ``browse``. Held on the shared registry
    /// so the SwiftUI approval dialog + the Settings auto-approve switch bind to
    /// the same object the tool runner consults.
    let browseApproval: BrowseApprovalStore
    /// Which backend ``web_search`` dispatches to + the stored API key. Owned by
    /// the registry so the chat loop doesn't need to thread a separate
    /// environment value through every tool call.
    let webSearch: WebSearchConfig
    /// Injected at the service boundary so the state transition can be tested
    /// without a live provider. Production always uses ``WebSearchTool/run``.
    private let webSearchRunner: WebSearchRunner

    init(
        browseApproval: BrowseApprovalStore = BrowseApprovalStore(),
        webSearch: WebSearchConfig = WebSearchConfig(),
        webSearchRunner: @escaping WebSearchRunner = { arguments, provider, apiKey in
            await WebSearchTool.run(
                arguments: arguments,
                provider: provider,
                apiKey: apiKey
            )
        }
    ) {
        self.browseApproval = browseApproval
        self.webSearch = webSearch
        self.webSearchRunner = webSearchRunner
    }

    var definitions: [ToolDefinition] {
        [
            WebSearchTool.definition,
            BrowseTool.definition,
            WeatherTool.definition,
        ]
    }

    func run(_ call: ToolCall) async -> ToolCallResult {
        let result: ToolCallResult
        switch call.function.name {
        case "web_search":
            result = await runWebSearchWithCredentialRecovery(call)
        case "browse":
            result = await BrowseTool.run(
                arguments: call.function.arguments,
                approval: browseApproval
            )
        case "weather":
            result = await WeatherTool.run(arguments: call.function.arguments)
        default:
            // The model invented a tool name we don't ship — return an
            // error result so it gets a chance to recover instead of
            // throwing and tearing the chat loop down.
            result = ToolCallResult(
                toolCallID: call.id,
                content: "unknown tool '\(call.function.name)' — available: web_search, browse, weather",
                isError: true
            )
        }
        // The individual tools don't know the toolCallID at run time, so
        // fill it in here. Classification is centralised at this boundary:
        // raw content continues to the model, but the transcript gets only a
        // stable diagnosis.
        let failureKind = result.failureKind ?? FailureDiagnoser.toolFailureKind(
            toolName: call.function.name,
            content: result.content,
            isError: result.isError
        )
        return ToolCallResult(
            toolCallID: call.id,
            content: result.content,
            isError: result.isError || failureKind != nil,
            failureKind: failureKind
        )
    }

    /// Execute one web-search call, with one narrowly-scoped configuration
    /// recovery: a rejected optional Keenable credential can transition to the
    /// provider's supported keyless mode and replay the SAME call once.
    ///
    /// This is not a generic retry loop. Producer-owned failure metadata is the
    /// gate, so network failures, quota/rate limits, malformed queries, and
    /// prose that happens to mention a key never enter this path. The rejected
    /// key is removed before replay, which makes resending it impossible and
    /// persists the selected recovery for later searches. If Keychain cannot
    /// establish that post-condition, the original failure remains visible.
    private func runWebSearchWithCredentialRecovery(
        _ call: ToolCall
    ) async -> ToolCallResult {
        let provider = webSearch.provider
        let key = webSearch.apiKey(for: provider)
        let first = await webSearchRunner(call.function.arguments, provider, key)

        guard provider == .keenable,
              key != nil,
              first.failureKind == .webSearchKeyRejected,
              !Task.isCancelled,
              webSearch.setAPIKey(nil, for: provider),
              !Task.isCancelled
        else {
            return first
        }

        let recovered = await webSearchRunner(call.function.arguments, provider, nil)
        return ToolCallResult(
            toolCallID: recovered.toolCallID,
            content: recovered.content + "\n\n" + Self.rejectedKeyRecoveryNote,
            isError: recovered.isError,
            failureKind: recovered.failureKind
        )
    }
}
