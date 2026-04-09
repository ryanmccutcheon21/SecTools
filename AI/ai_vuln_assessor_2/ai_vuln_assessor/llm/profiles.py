PROFILES = {
    "prompt_injection": ["pi_basic_override", "pi_translation_reframe", "pi_summary_repeat", "spe_direct"],
    "general_llm": ["pi_basic_override", "spe_direct", "ioh_xss", "hall_fake_paper"],
    "full": ["pi_basic_override", "pi_translation_reframe", "pi_summary_repeat", "spe_direct", "ioh_xss", "ioh_cmd", "hall_fake_paper"],
}
