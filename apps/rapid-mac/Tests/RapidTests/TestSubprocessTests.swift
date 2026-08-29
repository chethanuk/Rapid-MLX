import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Bounded async test subprocesses", TestTimeouts.hangProne)
struct TestSubprocessTests {
    private static var testRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    }

    @Test("captures both pipes without blocking the cooperative executor")
    func capturesOutputAndExit() async throws {
        let result = try await TestSubprocess.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "printf out; printf err >&2; exit 7"]
        )

        #expect(result.terminationStatus == 7)
        #expect(String(decoding: result.standardOutput, as: UTF8.self) == "out")
        #expect(String(decoding: result.standardError, as: UTF8.self) == "err")
    }

    @Test("a stalled child is sampled, killed, reaped, and reported within a bound")
    func stalledChildFailsWithinBound() async throws {
        let clock = ContinuousClock()
        let start = clock.now
        var childPID: pid_t?

        do {
            let result = try await TestSubprocess.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "printf '%s\\n' $$; exec sleep 30"],
                timeout: 0.2,
                sampleOnTimeout: true,
                sampleDuration: 1
            )
            childPID = pid_t(String(decoding: result.standardOutput, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines))
            Issue.record("expected the stalled child to time out")
        } catch let error as TestSubprocessError {
            guard case let .timedOut(_, seconds, pid) = error else {
                Issue.record("unexpected subprocess error: \(error)")
                return
            }
            childPID = pid
            #expect(seconds == 0.2)
            #expect(error.description.contains("process sample was emitted above"))
        }

        #expect(clock.now - start < .seconds(5))
        let pid = try #require(childPID)
        #expect(Darwin.kill(pid, 0) == -1)
        #expect(errno == ESRCH)
    }

    @Test("cancelling the caller terminates and reaps the child")
    func cancellationReapsChild() async throws {
        let task = Task {
            try await TestSubprocess.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "printf '%s\\n' $$; exec sleep 30"],
                timeout: 30,
                sampleOnTimeout: false
            )
        }

        try await Task.sleep(for: .milliseconds(100))
        task.cancel()

        do {
            _ = try await task.value
            Issue.record("cancelled subprocess unexpectedly returned a result")
        } catch is CancellationError {
            // Expected: cancellation reaches the caller only after the child
            // has exited and both pipe readers have observed EOF.
        }
    }

    @Test("blocking Process waits stay confined to native watchdog threads")
    func noBlockingWaitsInOrdinaryTests() throws {
        let allowed = Set([
            "Support/CIHangWatchdog.swift",
            "Support/TestSubprocess.swift",
            "TestSubprocessTests.swift", // Owns this source guard.
        ])
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: Self.testRoot,
                includingPropertiesForKeys: nil
            )
        )
        var offenders: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let relative = String(url.path.dropFirst(Self.testRoot.path.count + 1))
            guard !allowed.contains(relative) else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            if source.contains(".waitUntilExit()") { offenders.append(relative) }
        }

        #expect(
            offenders.isEmpty,
            "ordinary Swift tests must use TestSubprocess instead of parking a cooperative executor: \(offenders.sorted())"
        )
    }
}
