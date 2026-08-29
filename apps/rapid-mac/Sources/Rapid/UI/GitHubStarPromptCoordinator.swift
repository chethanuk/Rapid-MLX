import AppKit
import Foundation
import Observation

/// Owns the local-only, post-value invitation to visit Rapid-MLX on GitHub.
///
/// The prompt is deliberately workload-based rather than launch-based: it is
/// eligible only after successful product outcomes, waits for a quiet window,
/// and backs off exponentially when the user asks to see it later.
@MainActor
@Observable
final class GitHubStarPromptCoordinator {
    struct PresentationContext: Equatable {
        var isBusy: Bool
        var hasBlockingSurface: Bool

        static let ready = PresentationContext(isBusy: false, hasBlockingSurface: false)
    }

    enum Keys {
        static let completed = "Rapid.githubStar.completed.v1"
        static let totalSuccessfulActions = "Rapid.githubStar.totalSuccessfulActions.v1"
        static let baselineSuccessfulActions = "Rapid.githubStar.baselineSuccessfulActions.v1"
        static let nextWorkloadThreshold = "Rapid.githubStar.nextWorkloadThreshold.v1"
        static let deferredUntil = "Rapid.githubStar.deferredUntil.v1"
    }

    static let initialWorkloadThreshold = 35
    static let deferralInterval: TimeInterval = 3 * 24 * 60 * 60
    static let quietWindow: Duration = .milliseconds(1_200)

    private(set) var isPresented = false

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let now: () -> Date
    @ObservationIgnored private let quietWindow: Duration
    @ObservationIgnored private var context: PresentationContext = .ready
    @ObservationIgnored private var presentationPending = false
    @ObservationIgnored private var presentationTask: Task<Void, Never>?
    @ObservationIgnored private var keyboardMonitor: Any?

    init(
        defaults: UserDefaults = .standard,
        now: @escaping () -> Date = Date.init,
        quietWindow: Duration = GitHubStarPromptCoordinator.quietWindow
    ) {
        self.defaults = defaults
        self.now = now
        self.quietWindow = quietWindow
    }

    /// Records one delivered chat reply, dictation transcript, or generated
    /// image. No prompt state, content, or GitHub account data leaves the Mac.
    func productValueDelivered(_ kind: ProductValueKind) {
        guard !defaults.bool(forKey: Keys.completed) else { return }

        let previousTotal = defaults.integer(forKey: Keys.totalSuccessfulActions)
        let total = previousTotal == Int.max ? Int.max : previousTotal + 1
        defaults.set(total, forKey: Keys.totalSuccessfulActions)

        guard !isPresented, !presentationPending, isWorkloadEligible(total: total) else { return }
        presentationPending = true
        schedulePresentationCheck()
    }

    /// Keeps the prompt out of active generation and competing dialogs. A
    /// pending invitation is retained and retried once the workspace is calm.
    func updatePresentationContext(_ newContext: PresentationContext) {
        guard context != newContext else { return }
        context = newContext
        if newContext.isBusy || newContext.hasBlockingSurface {
            presentationTask?.cancel()
            presentationTask = nil
            if isPresented {
                isPresented = false
                presentationPending = true
            }
        } else if presentationPending {
            schedulePresentationCheck()
        }
    }

    /// Restarts the quiet window after a local key press without consuming or
    /// intercepting the event. This prevents the card arriving mid-sentence.
    func noteUserActivity() {
        guard presentationPending, !isPresented else { return }
        schedulePresentationCheck()
    }

    func startMonitoringUserActivity() {
        guard keyboardMonitor == nil else { return }
        keyboardMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { [weak self] event in
            self?.noteUserActivity()
            return event
        }
    }

    func stopMonitoringUserActivity() {
        guard let keyboardMonitor else { return }
        NSEvent.removeMonitor(keyboardMonitor)
        self.keyboardMonitor = nil
    }

    /// Close and Later are intentionally equivalent: both respect the user's
    /// timing, establish a three-day floor, and double the next workload gate.
    func deferPrompt() {
        guard isPresented else { return }
        isPresented = false
        presentationPending = false

        let total = defaults.integer(forKey: Keys.totalSuccessfulActions)
        let currentThreshold = workloadThreshold
        let (doubled, overflow) = currentThreshold.multipliedReportingOverflow(by: 2)
        defaults.set(total, forKey: Keys.baselineSuccessfulActions)
        defaults.set(overflow ? Int.max : doubled, forKey: Keys.nextWorkloadThreshold)
        defaults.set(now().addingTimeInterval(Self.deferralInterval), forKey: Keys.deferredUntil)
    }

    /// A successfully handed-off browser open completes this invitation for
    /// the install. Rapid does not probe the user's GitHub account to verify it.
    func repositoryOpened() {
        guard isPresented else { return }
        isPresented = false
        presentationPending = false
        defaults.set(true, forKey: Keys.completed)
    }

    private var workloadThreshold: Int {
        let stored = defaults.integer(forKey: Keys.nextWorkloadThreshold)
        return stored > 0 ? stored : Self.initialWorkloadThreshold
    }

    private func isWorkloadEligible(total: Int) -> Bool {
        let baseline = defaults.integer(forKey: Keys.baselineSuccessfulActions)
        let completedWork = max(0, total - baseline)
        guard completedWork >= workloadThreshold else { return false }
        guard let deferredUntil = defaults.object(forKey: Keys.deferredUntil) as? Date else { return true }
        return now() >= deferredUntil
    }

    private func schedulePresentationCheck() {
        presentationTask?.cancel()
        guard presentationPending, !context.isBusy, !context.hasBlockingSurface else {
            presentationTask = nil
            return
        }

        if quietWindow == .zero {
            presentIfReady()
            return
        }

        presentationTask = Task { [weak self, quietWindow] in
            try? await Task.sleep(for: quietWindow)
            guard !Task.isCancelled else { return }
            self?.presentIfReady()
        }
    }

    private func presentIfReady() {
        presentationTask = nil
        guard presentationPending,
              !isPresented,
              !context.isBusy,
              !context.hasBlockingSurface,
              !defaults.bool(forKey: Keys.completed),
              isWorkloadEligible(total: defaults.integer(forKey: Keys.totalSuccessfulActions)) else { return }
        presentationPending = false
        isPresented = true
    }
}
