import AppKit
import QuartzCore
import SwiftUI

/// Delivers frame callbacks in sync with the display containing this view.
struct StreamingPresentationDisplayLink: NSViewRepresentable {
    let isActive: Bool
    let onFrame: @MainActor (TimeInterval) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(isActive: isActive, onFrame: onFrame)
    }

    func makeNSView(context: Context) -> NSView {
        let view = NSView(frame: .zero)
        context.coordinator.attach(to: view)
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        context.coordinator.onFrame = onFrame
        context.coordinator.setActive(isActive)
        context.coordinator.attach(to: view)
    }

    static func dismantleNSView(_ view: NSView, coordinator: Coordinator) {
        coordinator.invalidate()
    }

    @MainActor
    final class Coordinator: NSObject {
        var onFrame: @MainActor (TimeInterval) -> Void
        private(set) var isActive: Bool
        private var displayLink: CADisplayLink?
        var isDisplayLinkPaused: Bool { displayLink?.isPaused ?? !isActive }

        init(
            isActive: Bool,
            onFrame: @escaping @MainActor (TimeInterval) -> Void
        ) {
            self.isActive = isActive
            self.onFrame = onFrame
        }

        func attach(to view: NSView) {
            guard displayLink == nil else { return }
            let link = view.displayLink(
                target: self,
                selector: #selector(displayLinkDidFire(_:))
            )
            link.isPaused = !isActive
            link.add(to: .main, forMode: .common)
            displayLink = link
        }

        func setActive(_ isActive: Bool) {
            self.isActive = isActive
            displayLink?.isPaused = !isActive
        }

        func invalidate() {
            displayLink?.invalidate()
            displayLink = nil
        }

        @objc private func displayLinkDidFire(_ link: CADisplayLink) {
            let duration = link.duration > 0
                ? link.duration
                : link.targetTimestamp - link.timestamp
            onFrame(duration)
        }
    }
}
