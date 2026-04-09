PROFILES = {
    "prompt_injection":["pi_basic_override","pi_delimiter","pi_token_smuggling","pi_translation_reframe","pi_summary_repeat","spe_direct","spe_indirect"],
    "system_prompt_leak":["spe_direct","spe_indirect","spe_completion","pi_summary_repeat","pi_translation_reframe"],
    "privacy":["pii_email","pii_credentials","pii_training_extract"],
    "general_llm":["pi_basic_override","pi_delimiter","jb_roleplay","spe_direct","ioh_xss","ioh_cmd","hall_fake_paper","dos_long"],
    "full":["pi_basic_override","pi_delimiter","pi_token_smuggling","pi_translation_reframe","pi_summary_repeat","jb_roleplay","jb_authority","spe_direct","spe_indirect","spe_completion","pii_email","pii_credentials","pii_training_extract","ioh_xss","ioh_sqli","ioh_cmd","hall_fake_paper","hall_future_event","dos_long","dos_recursive","scenario_delayed_leak"],
}
