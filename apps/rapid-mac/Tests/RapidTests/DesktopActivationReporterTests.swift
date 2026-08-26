import Foundation
import Testing
@testable import Rapid

@Suite("Desktop activation telemetry", .serialized)
struct DesktopActivationReporterTests {
    private func temporaryDirectory(_ label: String) -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("rapid-desktop-activation-\(label)-\(UUID().uuidString)")
    }

    private func event(_ kind: TelemetryEvent.Activation.Kind) -> TelemetryEvent {
        TelemetryEvent(
            schema_version: 1,
            client_id: "11111111-2222-3333-4444-555555555555",
            session_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            rapid_mlx_version: "0.13.2",
            event: .activation,
            timestamp: "2026-08-26T17:00:00Z",
            platform: .init(
                app: "rapid-desktop",
                os: "macos",
                os_version: "26.0.0",
                arch: "arm64"
            ),
            error_type: nil,
            error_message: nil,
            stack_frames: nil,
            context: nil,
            activation: .init(activation_kind: kind, surface: .desktop)
        )
    }

    @Test("Activation wire reuses the deployed two-field activation payload")
    func activationWireShape() throws {
        let data = try JSONEncoder().encode(event(.firstChatReply))
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["event"] as? String == "activation")
        let activation = try #require(json["activation"] as? [String: Any])
        #expect(activation["activation_kind"] as? String == "first_chat_reply")
        #expect(activation["surface"] as? String == "desktop")
        #expect(Set(activation.keys) == ["activation_kind", "surface"])
        #expect(json["request"] == nil)
        #expect(json["session"] == nil)
    }

    @Test("Declined consent touches no event, network, identity, marker, or directory")
    func disabledDoesNothing() async {
        let directory = temporaryDirectory("disabled")
        let probe = ActivationReporterProbe()
        let reporter = DesktopActivationReporter(
            isEnabled: { false },
            buildEvent: { kind in
                probe.didBuild()
                return event(kind)
            },
            sendEvent: { event in
                await probe.didSend(event)
                return true
            },
            markerDirectory: directory
        )

        await reporter.report(.firstChatReply)

        #expect(probe.buildCount == 0)
        #expect(await probe.sentCount == 0)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Accepted activation sends once and a new process observes its marker")
    func acceptedThenMarked() async throws {
        let directory = temporaryDirectory("accepted")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe()

        func makeReporter() -> DesktopActivationReporter {
            DesktopActivationReporter(
                isEnabled: { true },
                buildEvent: { event($0) },
                sendEvent: { event in
                    await probe.didSend(event)
                    return true
                },
                markerDirectory: directory
            )
        }

        let first = makeReporter()
        await first.report(.firstDictation)
        await first.report(.firstDictation)
        #expect(await probe.sentCount == 1)

        let second = makeReporter()
        await second.report(.firstDictation)
        #expect(await probe.sentCount == 1)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_dictation")
                    .path
            )
        )
    }

    @Test("Transport failure leaves the activation retryable")
    func failedSendRetries() async {
        let directory = temporaryDirectory("retry")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe(results: [false, true])
        let reporter = DesktopActivationReporter(
            isEnabled: { true },
            buildEvent: { event($0) },
            sendEvent: { event in await probe.nextResult(for: event) },
            markerDirectory: directory
        )

        await reporter.report(.firstImage)
        await reporter.report(.firstImage)

        #expect(await probe.sentCount == 2)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_image")
                    .path
            )
        )
    }
}

private final class ActivationReporterProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var builds = 0
    private var sent: [TelemetryEvent] = []
    private var results: [Bool]

    init(results: [Bool] = []) {
        self.results = results
    }

    var buildCount: Int {
        lock.withLock { builds }
    }

    var sentCount: Int {
        get async { lock.withLock { sent.count } }
    }

    func didBuild() {
        lock.withLock { builds += 1 }
    }

    func didSend(_ event: TelemetryEvent) async {
        lock.withLock { sent.append(event) }
    }

    func nextResult(for event: TelemetryEvent) async -> Bool {
        lock.withLock {
            sent.append(event)
            return results.isEmpty ? true : results.removeFirst()
        }
    }
}
