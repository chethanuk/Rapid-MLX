import Darwin
import Foundation

/// Sends the three product-approved Desktop activation milestones through the
/// existing telemetry envelope. Feature models publish a typed success; this
/// actor owns consent, once-per-install delivery, and marker persistence.
///
/// The order is intentional: consent -> marker lookup -> event construction ->
/// accepted send -> marker claim. An undecided/declined install therefore does
/// not create an identifier, event, marker, or pre-consent journal. A transport
/// failure leaves the milestone retryable, while a rare post-send marker failure
/// can only produce an at-least-once duplicate that the collector de-duplicates
/// by client ID.
actor DesktopActivationReporter {
    typealias Kind = TelemetryEvent.Activation.Kind
    typealias Enabled = @Sendable () -> Bool
    typealias BuildEvent = @Sendable (Kind) -> TelemetryEvent
    typealias SendEvent = @Sendable (TelemetryEvent) async -> Bool

    static let shared = DesktopActivationReporter()

    private let isEnabled: Enabled
    private let buildEvent: BuildEvent
    private let sendEvent: SendEvent
    private let markerDirectory: URL
    private var resolvedThisProcess: Set<Kind> = []
    private var inFlight: Set<Kind> = []

    init(
        isEnabled: @escaping Enabled = { TelemetryConfig.isEnabled },
        buildEvent: @escaping BuildEvent = { kind in
            TelemetryEvent.activation(
                version: TelemetryClient.currentVersion(),
                platform: TelemetryClient.currentPlatform(),
                kind: kind
            )
        },
        sendEvent: @escaping SendEvent = { event in
            await TelemetryClient().sendBatch([event])
        },
        markerDirectory: URL = TelemetryIdentity.sharedTelemetryDirectory()
    ) {
        self.isEnabled = isEnabled
        self.buildEvent = buildEvent
        self.sendEvent = sendEvent
        self.markerDirectory = markerDirectory
    }

    func report(_ kind: Kind) async {
        guard !resolvedThisProcess.contains(kind), !inFlight.contains(kind) else { return }

        // This is the privacy boundary. Do not even inspect the marker path
        // until the durable shared consent says telemetry is enabled.
        guard isEnabled() else { return }

        let marker = markerURL(for: kind)
        if FileManager.default.fileExists(atPath: marker.path) {
            resolvedThisProcess.insert(kind)
            return
        }

        inFlight.insert(kind)
        let event = buildEvent(kind)
        let accepted = await sendEvent(event)
        inFlight.remove(kind)
        // ``TelemetryClient.sendBatch`` deliberately reports opted-out as a
        // cleanup-success to its crash-marker caller. Re-check here before
        // claiming an activation marker so a Settings opt-out racing this send
        // can never retire a milestone that did not actually leave the Mac.
        // If consent changed after a real accepted send, leaving the marker
        // retryable permits only a harmless DISTINCT-client deduplicated replay.
        guard accepted, isEnabled() else { return }

        _ = claimMarker(at: marker)
        resolvedThisProcess.insert(kind)
    }

    private func markerURL(for kind: Kind) -> URL {
        markerDirectory.appendingPathComponent(
            "activation_seen_desktop_\(kind.rawValue)",
            isDirectory: false
        )
    }

    /// Best-effort O_EXCL claim, matching the engine activation marker
    /// primitive. The kind is a closed enum, so no user-controlled path
    /// component can escape the shared telemetry directory.
    private func claimMarker(at url: URL) -> Bool {
        do {
            try FileManager.default.createDirectory(
                at: markerDirectory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            return false
        }

        let descriptor = url.path.withCString {
            open($0, O_WRONLY | O_CREAT | O_EXCL, mode_t(0o600))
        }
        guard descriptor >= 0 else { return false }
        close(descriptor)
        return true
    }
}
