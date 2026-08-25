import Foundation
import Testing
@testable import Rapid

private final class SettingsKeychainProbe: KeychainStoring, @unchecked Sendable {
    private let lock = NSLock()
    private var reads = 0
    var result: KeychainReadResult = .missing

    var readCount: Int { lock.withLock { reads } }

    func read(account: String) -> String? { nil }
    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        lock.withLock { reads += 1 }
        return result
    }
    func write(account: String, secret: String) -> Bool { true }
    func delete(account: String) -> Bool { true }
}

@MainActor
@Suite("Settings web-search key draft commit")
struct SettingsWebSearchKeyDraftTests {
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test("Constructing the Tools configuration never reads Keychain")
    func constructionIsKeychainLazy() {
        let keychain = SettingsKeychainProbe()
        _ = WebSearchConfig(defaults: .standard, keychain: keychain)
        #expect(keychain.readCount == 0)
    }

    @Test("An explicit lazy probe preserves the denied state for inline recovery")
    func deniedProbeBecomesInlineState() async {
        let keychain = SettingsKeychainProbe()
        keychain.result = .unavailable
        let config = WebSearchConfig(defaults: .standard, keychain: keychain)

        #expect(config.cachedKeyState(for: .parallel) == .unknown)
        await config.prefetchAPIKey(for: .parallel)

        #expect(keychain.readCount == 1)
        #expect(config.cachedKeyState(for: .parallel) == .unavailable)
        #expect(config.apiKey(for: .parallel) == nil)
        #expect(keychain.readCount == 1, "the denied result is cached instead of repeatedly consulting Security.framework")
    }

    @Test("Tools page has no appearance-time Keychain read and wires only user-driven probes")
    func toolsPageUsesLazyReadTriggers() throws {
        let panel = try String(
            contentsOf: Self.packageRoot.appendingPathComponent("Sources/Rapid/UI/SettingsToolsPanel.swift"),
            encoding: .utf8
        )
        #expect(!panel.contains("prefetchAllAPIKeys"))
        #expect(panel.contains("guard provider.requiresKey else { return }"))
        #expect(panel.contains("focusedKeyProvider"))
        #expect(panel.contains("prefetchAPIKey(for: provider)"))

        let store = try String(
            contentsOf: Self.packageRoot.appendingPathComponent("Sources/Rapid/Tools/KeychainStore.swift"),
            encoding: .utf8
        )
        #expect(store.contains("kSecUseAuthenticationUISkip"))
        #expect(store.contains("legacyService).development"))
    }

    @Test("Untouched empty SecureField does not clear an existing stored key")
    func untouchedDraftIsUnchanged() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "", wasEdited: false) == .unchanged)
    }

    @Test("Edited whitespace draft clears the stored key")
    func editedWhitespaceClears() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "  \n\t ", wasEdited: true) == .clear)
    }

    @Test("Edited key trims before saving")
    func editedKeyTrimsBeforeSave() {
        #expect(SettingsView.webSearchKeyCommitAction(draft: "  BSA-key\n", wasEdited: true) == .save("BSA-key"))
    }

    // v0.6.7 codex r1 P2 — a failed Keychain write shows a
    // "Couldn't save, try again" banner; if the SecureField draft
    // is wiped at the same time the user has nothing to retry with
    // (the SecureField never echoes the existing key back, and the
    // Save button is gated on the dirty flag). Pin both branches.

    @Test("Successful write resets the draft + dirty flag")
    func successfulWriteResetsDraft() {
        #expect(SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: true))
    }

    @Test("Failed write keeps the draft so the user can retry without re-pasting")
    func failedWriteKeepsDraftForRetry() {
        #expect(!SettingsView.shouldResetWebSearchKeyDraftAfterCommit(keychainWriteSucceeded: false),
                "Without this, the 'try again' advice in the banner is impossible to follow — the SecureField never echoes the existing key, so the retry has nothing to commit.")
    }
}

/// v0.6.7 — pins the Save-button feedback contract. The transient
/// banner in Settings → Web Search reads its state off
/// ``SettingsView.WebSearchKeySaveFeedback``; the cases must remain
/// distinguishable by generation so back-to-back identical-outcome
/// Saves still retrigger the auto-dismiss task.
@MainActor
@Suite("Settings web-search key Save feedback")
struct SettingsWebSearchKeySaveFeedbackTests {
    @Test("Same kind with different generations compares non-equal")
    func sameKindBumpsViaGeneration() {
        let a = SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
        let b = SettingsView.WebSearchKeySaveFeedback.saved(generation: 2)
        #expect(a != b,
                "Without the generation bump, SwiftUI .task(id:) would see no change between back-to-back Saves and the auto-dismiss timer would never reschedule.")
    }

    @Test("Distinct kinds compare non-equal regardless of generation")
    func kindsAreDistinct() {
        #expect(SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.cleared(generation: 1))
        #expect(SettingsView.WebSearchKeySaveFeedback.saved(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 1))
        #expect(SettingsView.WebSearchKeySaveFeedback.cleared(generation: 1)
                != SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 1))
    }

    @Test("Same kind + same generation compares equal (idempotent)")
    func identicalEntriesAreEqual() {
        let a = SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 7)
        let b = SettingsView.WebSearchKeySaveFeedback.writeFailed(generation: 7)
        #expect(a == b)
    }
}
