import Foundation
import Security

/// Minimal Keychain wrapper for storing per-provider API keys.
///
/// We deliberately model the surface as a protocol so the test
/// suite can swap in an in-memory implementation rather than
/// touching the real system Keychain (which would prompt on
/// access, leak across test runs, and require manual cleanup).
protocol KeychainStoring: Sendable {
    func read(account: String) -> String?
    func readWithoutUserInteraction(account: String) -> KeychainReadResult
    @discardableResult func write(account: String, secret: String) -> Bool
    @discardableResult func delete(account: String) -> Bool
}

enum KeychainReadResult: Equatable, Sendable {
    case found(String)
    case missing
    case unavailable
}

extension KeychainStoring {
    /// Test doubles and non-system stores do not have an authentication UI.
    /// The real Keychain implementation overrides this with a query that
    /// explicitly forbids macOS from presenting one.
    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        if let value = read(account: account) { return .found(value) }
        return .missing
    }
}

/// Real-system implementation. Each entry is a ``kSecClassGenericPassword``
/// keyed by ``service = SystemKeychain.service`` + ``account``. We
/// use the generic-password class (not internet-password) because
/// Brave/Tavily keys are static credentials, not per-URL secrets.
///
/// Codex audit batch 6 finding (KeychainStore.swift:63, P2):
/// access policy is ``kSecAttrAccessibleWhenUnlockedThisDeviceOnly``.
/// The pre-audit shape used ``kSecAttrAccessibleAfterFirstUnlock``,
/// which (a) makes the key readable while the machine is locked
/// after the user's first post-boot login (any background process
/// running under the user account can read it) and (b) allows the
/// secret to be migrated off-device via Keychain sync / Time
/// Machine restore. ``WhenUnlockedThisDeviceOnly`` keeps the secret
/// readable only while the screen is unlocked and only on the
/// originating Mac.
struct SystemKeychain: KeychainStoring {
    /// The original unscoped service is read-only migration input. Local
    /// ad-hoc builds used the same service as notarized releases, so an item
    /// they created could carry an ACL that did not trust the release binary.
    private static let legacyService = "com.rapidmlx.rapid.api-keys"

    /// Namespace new items by the signing team. Notarized builds from the same
    /// team keep a stable service across releases; ad-hoc developer builds use
    /// an isolated namespace and can no longer create an item that later asks
    /// the release app for the login-keychain password.
    private static let teamIdentifier = currentTeamIdentifier()

    static let service: String = {
        if let teamIdentifier {
            return "\(legacyService).\(teamIdentifier)"
        }
        return "\(legacyService).development"
    }()

    func read(account: String) -> String? {
        guard case .found(let value) = readWithoutUserInteraction(account: account) else {
            return nil
        }
        return value.isEmpty ? nil : value
    }

    func readWithoutUserInteraction(account: String) -> KeychainReadResult {
        let current = query(account: account, service: Self.service)
        switch current {
        case .found:
            // An empty current-team item is a tombstone created by delete().
            // It deliberately masks a legacy value that this identity may not
            // be able to remove without showing a system authorization dialog.
            return current
        case .unavailable:
            return .unavailable
        case .missing:
            // Only a signed release identity may inspect the historical
            // shared namespace. Development builds must never touch it.
            guard Self.teamIdentifier != nil else { return .missing }
            return query(account: account, service: Self.legacyService)
        }
    }

    private func query(account: String, service: String) -> KeychainReadResult {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
            // A credential lookup must never summon a system password dialog.
            // Settings presents an inline recovery state instead.
            kSecUseAuthenticationUI as String: kSecUseAuthenticationUISkip,
        ]
        var item: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return .missing }
        guard status == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return .unavailable
        }
        return .found(value)
    }

    @discardableResult
    func write(account: String, secret: String) -> Bool {
        guard let data = secret.data(using: .utf8) else { return false }
        let baseQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account,
        ]
        // Try update first; if there's no existing item, fall
        // through to add. This is the canonical pattern for
        // "upsert" against the Keychain API. Update also bumps
        // the accessibility class so a pre-existing item written
        // with the prior (weaker) policy migrates forward on the
        // first write.
        let updateAttrs: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        var updateQuery = baseQuery
        updateQuery[kSecUseAuthenticationUI as String] = kSecUseAuthenticationUISkip
        let updateStatus = SecItemUpdate(updateQuery as CFDictionary, updateAttrs as CFDictionary)
        if updateStatus == errSecSuccess { return true }
        if updateStatus != errSecItemNotFound { return false }

        var addQuery = baseQuery
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        return addStatus == errSecSuccess
    }

    @discardableResult
    func delete(account: String) -> Bool {
        // Store an empty current-team item instead of deleting outright. It is
        // a non-secret tombstone that prevents a legacy ACL-mismatched value
        // from resurfacing on the next launch, and it can be written without
        // asking the user to authorize access to that legacy item.
        write(account: account, secret: "")
    }

    private static func currentTeamIdentifier() -> String? {
        guard let executableURL = Bundle.main.executableURL else { return nil }
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(executableURL as CFURL, [], &code) == errSecSuccess,
              let code else { return nil }
        var signingInfo: CFDictionary?
        guard SecCodeCopySigningInformation(code, SecCSFlags(rawValue: kSecCSSigningInformation), &signingInfo) == errSecSuccess,
              let info = signingInfo as? [String: Any],
              let team = info[kSecCodeInfoTeamIdentifier as String] as? String,
              !team.isEmpty else {
            return nil
        }
        return team
    }
}

/// In-memory backing for tests. Same surface as ``SystemKeychain``
/// but everything lives in a dictionary that dies with the
/// instance — no system-Keychain side effects, no popups, no
/// cross-test pollution. Thread-safe via a serial DispatchQueue
/// because the tool dispatcher may call into it from background
/// actor hops.
final class InMemoryKeychain: KeychainStoring, @unchecked Sendable {
    private var storage: [String: String] = [:]
    private let queue = DispatchQueue(label: "rapid.in-memory-keychain")

    func read(account: String) -> String? {
        queue.sync { storage[account] }
    }

    @discardableResult
    func write(account: String, secret: String) -> Bool {
        queue.sync { storage[account] = secret }
        return true
    }

    @discardableResult
    func delete(account: String) -> Bool {
        queue.sync { _ = storage.removeValue(forKey: account) }
        return true
    }
}
