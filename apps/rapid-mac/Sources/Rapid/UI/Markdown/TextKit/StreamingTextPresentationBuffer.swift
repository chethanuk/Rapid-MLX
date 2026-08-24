import Foundation

/// Separates uneven transport delivery from the text cadence shown on screen.
///
/// The buffer receives the full accumulated response, converts it to a suffix,
/// then releases complete extended grapheme clusters on display frames. Its
/// release rate only increases while a backlog exists, so a burst drains by
/// the latency target instead of slowing down as the queue gets shorter.
struct StreamingTextPresentationBuffer: Equatable {
    struct Configuration: Equatable {
        var targetLatency: TimeInterval = 0.12
        var completionDrainDuration: TimeInterval = 0.15
    }

    enum ReceiveResult: Equatable {
        case unchanged
        case appended
        case reset
    }

    private(set) var receivedText = ""
    private(set) var presentedText = ""
    private(set) var pendingText = ""
    private(set) var pendingGraphemeCount = 0

    private var graphemesPerSecond: Double = 0
    private let configuration: Configuration

    init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    var hasPendingText: Bool { pendingGraphemeCount > 0 }

    /// Accept the transport's full accumulated text without scanning its
    /// growing prefix. A non-monotonic update is treated as a replacement.
    @discardableResult
    mutating func receive(_ text: String) -> ReceiveResult {
        let previousByteCount = receivedText.utf8.count
        let nextByteCount = text.utf8.count

        if nextByteCount < previousByteCount
            || (nextByteCount == previousByteCount && text != receivedText) {
            receivedText = text
            presentedText = ""
            pendingText = text
            pendingGraphemeCount = text.count
            graphemesPerSecond = 0
            updateReleaseRate(targetDuration: configuration.targetLatency)
            return .reset
        }
        guard nextByteCount > previousByteCount else { return .unchanged }

        let utf8 = text.utf8
        let suffixStart = utf8.index(utf8.startIndex, offsetBy: previousByteCount)
        guard let suffix = String(utf8[suffixStart...]) else {
            receivedText = text
            presentedText = ""
            pendingText = text
            pendingGraphemeCount = text.count
            graphemesPerSecond = 0
            updateReleaseRate(targetDuration: configuration.targetLatency)
            return .reset
        }

        receivedText = text
        pendingText.append(suffix)
        pendingGraphemeCount += suffix.count
        updateReleaseRate(targetDuration: configuration.targetLatency)
        return .appended
    }

    /// Return the delta that should become visible on this display frame.
    mutating func presentFrame(
        duration: TimeInterval,
        isFinishing: Bool = false
    ) -> String? {
        guard pendingGraphemeCount > 0 else { return nil }

        let targetDuration = isFinishing
            ? configuration.completionDrainDuration
            : configuration.targetLatency
        updateReleaseRate(targetDuration: targetDuration)

        let safeDuration = duration.isFinite && duration > 0 ? duration : 1.0 / 60.0
        let releaseCount = min(
            pendingGraphemeCount,
            max(1, Int(ceil(graphemesPerSecond * safeDuration)))
        )
        let end = pendingText.index(
            pendingText.startIndex,
            offsetBy: releaseCount
        )
        let delta = String(pendingText[..<end])
        pendingText.removeSubrange(..<end)
        pendingGraphemeCount -= releaseCount
        presentedText.append(delta)

        if pendingGraphemeCount == 0 {
            graphemesPerSecond = 0
        }
        return delta
    }

    /// Correctness fallback used when a new response starts before the prior
    /// response's short completion drain has elapsed.
    mutating func drainAll() -> String? {
        guard !pendingText.isEmpty else { return nil }
        let delta = pendingText
        presentedText.append(delta)
        pendingText = ""
        pendingGraphemeCount = 0
        graphemesPerSecond = 0
        return delta
    }

    private mutating func updateReleaseRate(targetDuration: TimeInterval) {
        guard pendingGraphemeCount > 0 else { return }
        let safeTarget = max(targetDuration, 1.0 / 240.0)
        graphemesPerSecond = max(
            graphemesPerSecond,
            Double(pendingGraphemeCount) / safeTarget
        )
    }
}
