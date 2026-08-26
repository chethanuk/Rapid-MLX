import Testing
@testable import Rapid

@MainActor
@Suite("Quickstart hardware-aware starter policy")
struct QuickstartStarterPolicyTests {
    private func hardware(_ ramGB: Double) -> MacHardware {
        MacHardware(
            brandString: "Test Mac",
            family: .m3,
            tier: .base,
            physicalRAMBytes: UInt64(ramGB * 1_073_741_824),
            memoryBandwidthGBs: 100
        )
    }

    private func entry(
        _ alias: String,
        cached: Bool = false,
        kind: ModelKind = .chat
    ) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "fixture/\(alias)",
            sizeOnDisk: nil,
            cached: cached,
            kind: kind
        )
    }

    @Test("RAM threshold chooses 2.6B below 16 GB and Qwen 4B at 16 GB or above")
    func ramMatrix() {
        let catalog = [
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit"),
        ]

        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(8), catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(15.99), catalog: catalog
        ).alias == "lfm2.5-2.6b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(16), catalog: catalog
        ).alias == "qwen3.5-4b-4bit")
        #expect(QuickstartCoordinator.defaultChoice(
            hardware: hardware(64), catalog: catalog
        ).alias == "qwen3.5-4b-4bit")
    }

    @Test("An eligible cached chat model wins without a download")
    func cachedEligibleWins() {
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: [
                entry("qwen3.5-4b-4bit"),
                entry("qwen3.5-9b-4bit", cached: true),
            ]
        )
        #expect(pick.alias == "qwen3.5-9b-4bit")
    }

    @Test("The chooser presents one hardware-fit starter, not two competing recommendations")
    func shortlistHasOneStarter() {
        let catalog = [
            entry("lfm2.5-2.6b-4bit"),
            entry("qwen3.5-4b-4bit"),
        ]

        let compact = QuickstartView.shortlist(
            catalog: catalog,
            selection: "lfm2.5-2.6b-4bit",
            physicalRAMGB: 8
        )
        #expect(compact.starters.map(\.alias) == ["lfm2.5-2.6b-4bit"])

        let standard = QuickstartView.shortlist(
            catalog: catalog,
            selection: "qwen3.5-4b-4bit",
            physicalRAMGB: 16
        )
        #expect(standard.starters.map(\.alias) == ["qwen3.5-4b-4bit"])
    }

    @Test("Cached media and the 1.2B escape never become automatic starters")
    func ineligibleCacheDoesNotWin() {
        let pick = QuickstartCoordinator.defaultChoice(
            hardware: hardware(16),
            catalog: [
                entry("qwen3.5-4b-4bit"),
                entry("lfm2.5-1b-4bit", cached: true),
                entry("flux-klein", cached: true, kind: .image),
            ]
        )
        #expect(pick.alias == "qwen3.5-4b-4bit")
        #expect(QuickstartCoordinator.lowMemoryChoice.alias == "lfm2.5-1b-4bit")
    }

    @Test("A later cache refresh never overrides an explicit user choice")
    func explicitSelectionWins() {
        let coordinator = QuickstartCoordinator()
        let explicit = QuickstartCoordinator.choice(forAlias: "qwen3.5-9b-4bit")
        coordinator.select(explicit)
        coordinator.applyDefaultChoice(
            hardware: hardware(8),
            catalog: [entry("lfm2.5-2.6b-4bit", cached: true)]
        )
        #expect(coordinator.selection.alias == explicit.alias)
    }
}
