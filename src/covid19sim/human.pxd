cdef extern from "native/_native.h":
    ctypedef class covid19sim.native._native.BaseHuman [object BaseHumanObject]:
        pass

cdef class Human(BaseHuman):
    cdef public:
        object env, city, recovered_timestamp, _infection_timestamp, exposure_source, rng, profession, work_start_time, work_end_time, infection_ratio, cold_timestamp, flu_timestamp, allergy_timestamp, last_state, test_type, test_time, viral_load_plateau_height, viral_load_plateau_start, viral_load_plateau_end, hidden_test_result, _will_report_test_result, time_to_test_result, test_result_validated, _test_results, denied_icu, denied_icu_days, covid_symptom_start_time, rolling_all_symptoms, rolling_all_reported_symptoms, obs_age, obs_sex, tracing_method, _maintain_extra_distance, _follows_recommendations_today, recommendations_to_follow, _time_encounter_reduction_factor, contact_book, proba_to_risk_level_map, household, location, visits, last_date, avg_shopping_time, scale_shopping_time, avg_exercise_time, scale_exercise_time, avg_working_minutes, scale_working_minutes, avg_misc_time, scale_misc_time, number_of_shopping_days, number_of_shopping_hours, number_of_exercise_days, number_of_exercise_hours, number_of_misc_hours, max_misc_per_week, max_exercise_per_week, max_shop_per_week, location_leaving_time, location_start_time, age_bin_width_5, shopping_days, shopping_hours, exercise_days, exercise_hours, misc_hours, work_start_hour, working_days, _workplace;
        object    name, sex;
        object   conf, infectiousness_history_map, risk_history_map, prev_risk_history_map;
        object    known_connections;
        object   my_history, r0, _events, preexisting_conditions, allergy_progression, cold_progression, flu_progression, all_symptoms, cold_symptoms, flu_symptoms, covid_symptoms, allergy_symptoms, time_slots, obs_preexisting_conditions, covid_progression, stores_preferences, parks_preferences;
        object    n_infectious_contacts, age, len_allergies, incubation_days, mean_daily_interaction_age_group, num_contacts, count_misc, count_exercise, count_shop;
        object inflammatory_disease_level, carefulness, normalized_susceptibility, initial_viral_load, phone_bluetooth_noise, mask_efficacy, _rec_level, _intervention_level, hygiene, effective_contacts, last_sent_update_gaen, rho, gamma, infectiousness_onset_days, viral_load_peak_height, viral_load_peak_start, recovery_days, peak_height, plateau_height, plateau_end_recovery_slope, peak_plateau_slope;
        object   is_asymptomatic, is_healthcare_worker, does_not_work, track_this_human, has_allergies, can_get_really_sick, can_get_extremely_sick, never_recovers, is_immune, has_app, has_logged_info, obs_is_healthcare_worker, obs_hospitalized, obs_in_icu, tracing, WEAR_MASK, wearing_mask, notified, _test_recommended, rest_at_home, travelled_recently, track_this_guy;
    
        # str    name, sex;
        # dict   conf, infectiousness_history_map, risk_history_map, prev_risk_history_map;
        # set    known_connections;
        # list   my_history, r0, _events, preexisting_conditions, allergy_progression, cold_progression, flu_progression, all_symptoms, cold_symptoms, flu_symptoms, covid_symptoms, allergy_symptoms, time_slots, obs_preexisting_conditions, covid_progression, stores_preferences, parks_preferences;
        # int    n_infectious_contacts, age, len_allergies, incubation_days, mean_daily_interaction_age_group, num_contacts, count_misc, count_exercise, count_shop;
        # double inflammatory_disease_level, carefulness, normalized_susceptibility, initial_viral_load, phone_bluetooth_noise, mask_efficacy, _rec_level, _intervention_level, hygiene, effective_contacts, last_sent_update_gaen, rho, gamma, infectiousness_onset_days, viral_load_peak_height, viral_load_peak_start, recovery_days, peak_height, plateau_height, plateau_end_recovery_slope, peak_plateau_slope;
        # bint   is_asymptomatic, is_healthcare_worker, does_not_work, track_this_human, has_allergies, can_get_really_sick, can_get_extremely_sick, never_recovers, is_immune, has_app, has_logged_info, obs_is_healthcare_worker, obs_hospitalized, obs_in_icu, tracing, WEAR_MASK, wearing_mask, notified, _test_recommended, rest_at_home, travelled_recently, track_this_guy;