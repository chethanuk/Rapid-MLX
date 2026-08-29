import Darwin
import Foundation

struct TestSubprocessResult: Sendable {
    let terminationStatus: Int32
    let standardOutput: Data
    let standardError: Data
}

enum TestSubprocessError: Error, CustomStringConvertible {
    case timedOut(command: String, seconds: TimeInterval, pid: pid_t)

    var description: String {
        switch self {
        case let .timedOut(command, seconds, pid):
            return "test subprocess timed out after \(seconds)s (pid \(pid)): \(command); process sample was emitted above"
        }
    }
}

/// Async, bounded launcher for test-only subprocesses.
///
/// A synchronous ``Process.waitUntilExit()`` occupies a cooperative executor
/// worker when called from an async or MainActor test. On a low-core runner,
/// enough such waits can prevent the tasks responsible for making the child
/// exit from running at all. This helper instead resumes from
/// ``terminationHandler`` and gives every child a watchdog on a native thread,
/// independent of Swift concurrency. A timeout samples the child into the
/// captured test log, then escalates TERM to KILL.
enum TestSubprocess {
    static func run(
        executableURL: URL,
        arguments: [String] = [],
        currentDirectoryURL: URL? = nil,
        environment: [String: String]? = nil,
        timeout: TimeInterval = 30,
        sampleOnTimeout: Bool = true,
        sampleDuration: Int = 3
    ) async throws -> TestSubprocessResult {
        precondition(timeout > 0, "test subprocess timeout must be positive")
        precondition(sampleDuration > 0, "sample duration must be positive")

        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectoryURL
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdoutCapture = AsyncPipeCapture(stdoutPipe.fileHandleForReading)
        let stderrCapture = AsyncPipeCapture(stderrPipe.fileHandleForReading)
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let state = TestProcessExitState()
        let processBox = UncheckedProcess(process)
        process.terminationHandler = { terminated in
            state.complete(status: terminated.terminationStatus)
        }

        do {
            try process.run()
        } catch {
            stdoutPipe.fileHandleForWriting.closeFile()
            stderrPipe.fileHandleForWriting.closeFile()
            stdoutCapture.cancel()
            stderrCapture.cancel()
            throw error
        }
        stdoutPipe.fileHandleForWriting.closeFile()
        stderrPipe.fileHandleForWriting.closeFile()

        let command = ([executableURL.path] + arguments).joined(separator: " ")
        let pid = process.processIdentifier
        Thread.detachNewThread {
            guard state.beginTimeoutIfIncomplete(after: timeout) else { return }
            if sampleOnTimeout {
                sample(pid: pid, duration: sampleDuration, command: command)
            }
            guard processBox.process.isRunning else { return }
            processBox.process.terminate()
            guard !state.waitForCompletion(seconds: 2) else { return }
            Darwin.kill(pid, SIGKILL)
        }

        let exit = await withTaskCancellationHandler {
            await state.value()
        } onCancel: {
            guard processBox.process.isRunning else { return }
            processBox.process.terminate()
            Thread.detachNewThread {
                guard !state.waitForCompletion(seconds: 2) else { return }
                Darwin.kill(pid, SIGKILL)
            }
        }
        async let standardOutput = stdoutCapture.value()
        async let standardError = stderrCapture.value()
        let output = await standardOutput
        let errorOutput = await standardError
        try Task.checkCancellation()
        if exit.timedOut {
            throw TestSubprocessError.timedOut(
                command: command,
                seconds: timeout,
                pid: pid
            )
        }
        return TestSubprocessResult(
            terminationStatus: exit.status,
            standardOutput: output,
            standardError: errorOutput
        )
    }

    private static func sample(pid: pid_t, duration: Int, command: String) {
        let message = "TestSubprocess: timeout; sampling pid \(pid) for \(duration)s: \(command)\n"
        FileHandle.standardError.write(Data(message.utf8))
        let sampler = Process()
        sampler.executableURL = URL(fileURLWithPath: "/usr/bin/sample")
        sampler.arguments = [String(pid), String(duration), "-file", "/dev/stderr"]
        sampler.standardOutput = FileHandle.standardError
        sampler.standardError = FileHandle.standardError
        do {
            try sampler.run()
            // This runs on the dedicated watchdog Thread, never a Swift
            // cooperative executor worker.
            sampler.waitUntilExit()
        } catch {
            let failure = "TestSubprocess: sample failed for pid \(pid): \(error)\n"
            FileHandle.standardError.write(Data(failure.utf8))
        }
    }
}

private final class UncheckedProcess: @unchecked Sendable {
    let process: Process

    init(_ process: Process) {
        self.process = process
    }
}

private final class TestProcessExitState: @unchecked Sendable {
    struct Exit: Sendable {
        let status: Int32
        let timedOut: Bool
    }

    private let condition = NSCondition()
    private var exit: Exit?
    private var timedOut = false
    private var continuation: CheckedContinuation<Exit, Never>?

    func complete(status: Int32) {
        condition.lock()
        guard exit == nil else {
            condition.unlock()
            return
        }
        let value = Exit(status: status, timedOut: timedOut)
        exit = value
        let continuation = continuation
        self.continuation = nil
        condition.broadcast()
        condition.unlock()
        continuation?.resume(returning: value)
    }

    func value() async -> Exit {
        await withCheckedContinuation { continuation in
            condition.lock()
            if let exit {
                condition.unlock()
                continuation.resume(returning: exit)
            } else {
                self.continuation = continuation
                condition.unlock()
            }
        }
    }

    func beginTimeoutIfIncomplete(after seconds: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: seconds)
        while exit == nil && condition.wait(until: deadline) {}
        guard exit == nil else { return false }
        timedOut = true
        return true
    }

    func waitForCompletion(seconds: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date(timeIntervalSinceNow: seconds)
        while exit == nil && condition.wait(until: deadline) {}
        return exit != nil
    }
}

private final class AsyncPipeCapture: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle
    private var data = Data()
    private var finished = false
    private var continuation: CheckedContinuation<Data, Never>?

    init(_ handle: FileHandle) {
        self.handle = handle
        handle.readabilityHandler = { [weak self] readable in
            guard let self else { return }
            let chunk = readable.availableData
            if chunk.isEmpty {
                finish()
            } else {
                lock.lock()
                data.append(chunk)
                lock.unlock()
            }
        }
    }

    func value() async -> Data {
        await withCheckedContinuation { continuation in
            lock.lock()
            if finished {
                let data = data
                lock.unlock()
                continuation.resume(returning: data)
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func cancel() {
        finish()
    }

    private func finish() {
        handle.readabilityHandler = nil
        lock.lock()
        guard !finished else {
            lock.unlock()
            return
        }
        finished = true
        let data = data
        let continuation = continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(returning: data)
    }
}
