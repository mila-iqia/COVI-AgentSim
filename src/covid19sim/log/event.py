import datetime
import logging
from covid19sim.epidemiology.viral_load import viral_load_for_day

class Event:
    """
    [summary]
    """
    test = 'test'
    encounter = 'encounter'
    encounter_message = 'encounter_message'
    risk_update = 'risk_update'
    contamination = 'contamination'
    recovered = 'recovered'
    static_info = 'static_info'
    visit = 'visit'
    daily = 'daily'

    @staticmethod
    def members():
        """
        DEPRECATED
        """
        return [Event.test, Event.encounter, Event.contamination, Event.static_info, Event.visit, Event.daily]

    @staticmethod
    def log_encounter(COLLECT_LOGS, human1, human2, location, duration, distance, infectee, p_infection, time):
        """
        Logs the encounter between `human1` and `human2` at `location` for `duration`
        while staying at `distance` from each other. If infectee is not None, it is
        either human1.name or human2.name.

        Each of the two humans gets its `events` attribute appended whit a dictionnary
        describing the encounter:

        human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter,
                'time':time,
                'payload':{
                    'observed': obs_payload,  # None if one of the humans does not have
                                              # the app. Otherwise contains the observed
                                              # data: lat, lon, location_type
                    'unobserved':unobs_payload  # unobserved data, see loc_unobs_keys and
                                                # h_unobs_keys
                }
        })


        Args:
            COLLECT_LOGS (bool): Log the event in a file if True
            human1 (covid19sim.human.Human): One of the encounter's 2 humans
            human2 (covid19sim.human.Human): One of the encounter's 2 humans
            location (covid19sim.city.Location): Where the encounter happened
            duration (int): duration of encounter
            distance (float): distance between people (TODO: meters? cm?)
            infectee (str | None): name of the human which is infected, if any.
                None otherwise
            p_infection (float): probability for the infectee of getting infected
            time (datetime.datetime): timestamp of encounter
        """
        if COLLECT_LOGS:
            h_obs_keys   = ['obs_lat', 'obs_lon']

            h_unobs_keys = ['carefulness', 'viral_load', 'infectiousness',
                            'symptoms', 'is_exposed', 'is_infectious',
                            'infection_timestamp', 'is_really_sick',
                            'is_extremely_sick', 'sex',  'wearing_mask', 'mask_efficacy',
                            'risk', 'risk_level', 'rec_level']

            loc_obs_keys = ['location_type', 'lat', 'lon']
            loc_unobs_keys = ['contamination_probability', 'social_contact_factor']

            obs, unobs = [], []

            same_household = (human1.household.name == human2.household.name) & (location.name == human1.household.name)
            for human in [human1, human2]:
                o = {key:getattr(human, key) for key in h_obs_keys}
                obs.append(o)
                u = {key:getattr(human, key) for key in h_unobs_keys}
                u['human_id'] = human.name
                u['location_is_residence'] = human.household == location
                u['got_exposed'] = infectee == human.name if infectee else False
                u['exposed_other'] = infectee != human.name if infectee else False
                u['same_household'] = same_household
                u['infectiousness_start_time'] = None if not u['got_exposed'] else human.infection_timestamp + datetime.timedelta(days=human.infectiousness_onset_days)
                u['obs_hospitalized'] = human.mobility_planner.hospitalization_timestamp is not None
                u['obs_in_icu'] = human.mobility_planner.critical_condition_timestamp is not None
                unobs.append(u)

            loc_obs = {key:getattr(location, key) for key in loc_obs_keys}
            loc_unobs = {key:getattr(location, key) for key in loc_unobs_keys}
            loc_unobs['location_p_infection'] = location.contamination_probability / location.social_contact_factor
            other_obs = {'duration':duration, 'distance':distance}
            other_unobs = {'p_infection':p_infection}
            both_have_app = human1.has_app and human2.has_app
            for i, human in [(0, human1), (1, human2)]:
                if both_have_app:
                    obs_payload = {**loc_obs, **other_obs, 'human1':obs[i], 'human2':obs[1-i]}
                    unobs_payload = {**loc_unobs, **other_unobs, 'human1':unobs[i], 'human2':unobs[1-i]}
                else:
                    obs_payload = {}
                    unobs_payload = { **loc_obs, **loc_unobs, **other_obs, **other_unobs, 'human1':{**obs[i], **unobs[i]},
                                        'human2': {**obs[1-i], **unobs[1-i]} }

                human.events.append({
                    'human_id':human.name,
                    'event_type':Event.encounter,
                    'time':time,
                    'payload':{'observed':obs_payload, 'unobserved':unobs_payload}
                })

        logging.info(f"{time} - {human1.name} and {human2.name} {Event.encounter} event")
        logging.debug("{time} - {human1.name}{h1_infectee} "
                      "(viral load:{h1_viralload:.3f}, risk:{h1_risk:.3f}, "
                      "risk lvl:{h1_risk_lvl}, rec lvl:{h1_rec_lvl}) "
                      "encountered {human2.name}{h2_infectee} "
                      "(viral load:{h2_viralload:.3f}, risk:{h2_risk:.3f}, "
                      "risk lvl:{h2_risk_lvl}, rec lvl:{h2_rec_lvl}) "
                      "for {duration:.2f}min at ({location.lat}, {location.lon}) "
                      "and stayed at {distance:.2f}cm. The probability of getting "
                      "infected was {p_infection:.6f}"
                      .format(time=time,
                              human1=human1,
                              h1_viralload=viral_load_for_day(human1, time),
                              h1_risk=human1.risk,
                              h1_risk_lvl=human1.risk_level,
                              h1_rec_lvl=human1.rec_level,
                              h1_infectee=' (infectee)' if infectee == human1.name else '',
                              human2=human2,
                              h2_viralload=viral_load_for_day(human2, time),
                              h2_risk=human2.risk,
                              h2_risk_lvl=human2.risk_level,
                              h2_rec_lvl=human2.rec_level,
                              h2_infectee=' (infectee)' if infectee == human2.name else '',
                              duration=duration,
                              location=location,
                              distance=distance,
                              p_infection=p_infection if p_infection else 0.0
                      ))

    def log_encounter_messages(COLLECT_LOGS, human1, human2, location, duration, distance, time):
        """
        Logs the encounter between `human1` and `human2` at `location` for `duration`
        while staying at `distance` from each other. If infectee is not None, it is
        either human1.name or human2.name.

        Each of the two humans gets its `events` attribute appended whit a dictionnary
        describing the encounter:

        human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter,
                'time':time,
                'payload':{
                    'observed': obs_payload,  # None if one of the humans does not have
                                              # the app. Otherwise contains the observed
                                              # data: lat, lon, location_type
                    'unobserved':unobs_payload  # unobserved data, see loc_unobs_keys and
                                                # h_unobs_keys
                }
        })


        Args:
            COLLECT_LOGS (bool): Log the event in a file if True
            human1 (covid19sim.human.Human): One of the encounter's 2 humans
            human2 (covid19sim.human.Human): One of the encounter's 2 humans
            location (covid19sim.city.Location): Where the encounter happened
            duration (int): duration of encounter
            distance (float): distance between people (TODO: meters? cm?)
            infectee (str | None): name of the human which is infected, if any.
                None otherwise
            time (datetime.datetime): timestamp of encounter
        """
        if COLLECT_LOGS:
            h_obs_keys   = ['obs_hospitalized', 'obs_in_icu',
                            'obs_lat', 'obs_lon']

            h_unobs_keys = ['carefulness', 'viral_load', 'infectiousness',
                            'symptoms', 'is_exposed', 'is_infectious',
                            'infection_timestamp', 'is_really_sick',
                            'is_extremely_sick', 'sex',  'wearing_mask', 'mask_efficacy',
                            'risk', 'risk_level', 'rec_level']

            loc_obs_keys = ['location_type', 'lat', 'lon']
            loc_unobs_keys = ['contamination_probability', 'social_contact_factor']

            obs, unobs = [], []

            same_household = (human1.household.name == human2.household.name) & (location.name == human1.household.name)
            for human in [human1, human2]:
                o = {key:getattr(human, key) for key in h_obs_keys}
                obs.append(o)
                u = {key:getattr(human, key) for key in h_unobs_keys}
                u['human_id'] = human.name
                u['location_is_residence'] = human.household == location
                u['same_household'] = same_household
                unobs.append(u)

            loc_obs = {key:getattr(location, key) for key in loc_obs_keys}
            loc_unobs = {key:getattr(location, key) for key in loc_unobs_keys}
            loc_unobs['location_p_infection'] = location.contamination_probability / location.social_contact_factor
            other_obs = {'duration':duration, 'distance':distance}
            both_have_app = human1.has_app and human2.has_app
            for i, human in [(0, human1), (1, human2)]:
                if both_have_app:
                    obs_payload = {**loc_obs, **other_obs, 'human1':obs[i], 'human2':obs[1-i]}
                    unobs_payload = {**loc_unobs, 'human1':unobs[i], 'human2':unobs[1-i]}
                else:
                    obs_payload = {}
                    unobs_payload = { **loc_obs, **loc_unobs, **other_obs, 'human1':{**obs[i], **unobs[i]},
                                        'human2': {**obs[1-i], **unobs[1-i]} }

            human.events.append({
                'human_id':human.name,
                'event_type':Event.encounter_message,
                'time':time,
                'payload':{'observed':obs_payload, 'unobserved':unobs_payload}
            })

        logging.info(f"{time} - {human1.name} and {human2.name} {Event.encounter_message} event")
        logging.debug("{time} - {human1.name} and {human2.name} exchanged encounter "
                      "messages for {duration:.2f}min at ({location.lat}, {location.lon}) "
                      "and stayed at {distance:.2f}cm"
                      .format(time=time,
                              human1=human1,
                              human2=human2,
                              duration=duration,
                              location=location,
                              distance=distance))

    def log_risk_update(COLLECT_LOGS, human, tracing_description,
                        prev_risk_history_map, risk_history_map, current_day_idx,
                        time):
        if COLLECT_LOGS:
            human.events.append({
                'human_id':human.name,
                'event_type':Event.risk_update,
                'time':time,
                'payload':{
                    'observed': {
                        'tracing_description': tracing_description,
                        'prev_risk_history_map': prev_risk_history_map,
                        'risk_history_map': risk_history_map,
                    }
                }
            })

        logging.info(f"{time} - {human.name} {Event.risk_update} event")
        logging.debug(f"{time} - {human.name} updated his risk from "
                      f"{prev_risk_history_map[current_day_idx]} to "
                      f"{risk_history_map[current_day_idx]} following "
                      f"{tracing_description} rules")

    @staticmethod
    def log_test(COLLECT_LOGS, human, time):
        """
        Adds an event to a human's `events` list if COLLECT_LOGS is True.
        Events contains the test resuts time, reported_test_result,
        reported_test_type, test_result_validated, test_type, test_result
        split across observed and unobserved data.

        Args:
            COLLECT_LOGS ([type]): [description]
            human (covid19sim.human.Human): Human whose test should be logged
            time (datetime.datetime): Event's time
        """
        if COLLECT_LOGS:
            human.events.append(
                {
                    'human_id': human.name,
                    'event_type': Event.test,
                    'time': time,
                    'payload': {
                        'observed':{
                            'result': human.reported_test_result,
                            'test_type':human.reported_test_type,
                            'validated_test_result':human.test_result_validated
                        },
                        'unobserved':{
                            'test_type':human.test_type,
                            'result': human.test_result
                        }

                    }
                }
            )
        logging.info(f"{time} - {human.name} {Event.test} event")
        logging.debug(f"{time} - {human.name} tested {human.test_result}.")

    @staticmethod
    def log_daily(COLLECT_LOGS, human, time):
        """
        Adds an event to a human's `events` list containing daily health information
        like symptoms, infectiousness and viral_load.

        Args:
            COLLECT_LOGS ([type]): [description]
            human (covid19sim.human.Human): Human who's health should be logged
            time (datetime.datetime): Event time
        """
        if COLLECT_LOGS:
            human.events.append(
                {
                    'human_id': human.name,
                    'event_type': Event.daily,
                    'time': time,
                    'payload': {
                        'observed':{
                            "reported_symptoms": human.all_reported_symptoms
                        },
                        'unobserved':{
                            'infectiousness': human.infectiousness,
                            "viral_load": human.viral_load,
                            "all_symptoms": human.all_symptoms,
                            "covid_symptoms":human.covid_symptoms,
                            "flu_symptoms":human.flu_symptoms,
                            "cold_symptoms":human.cold_symptoms
                        }
                    }
                }
            )
        logging.info(f"{time} - {human.name} {Event.daily} event")

    @staticmethod
    def log_exposed(COLLECT_LOGS, human, source, p_infection, time):
        """
        [summary]

        Args:
            COLLECT_LOGS ([type]): [description]
            human ([type]): [description]
            source ([type]): [description]
            p_infection (float): probability for the infectee of getting infected
            time ([type]): [description]
        """
        if COLLECT_LOGS:
            human.events.append(
                {
                    'human_id': human.name,
                    'event_type': Event.contamination,
                    'time': time,
                    'payload': {
                        'observed':{
                        },
                        'unobserved':{
                          'exposed': True,
                          'source':source.name,
                          'source_is_location': 'human' not in source.name,
                          'source_is_human': 'human' in source.name,
                          'infectiousness_start_time': human.infection_timestamp + datetime.timedelta(days=human.infectiousness_onset_days),
                          'p_infection': p_infection
                        }
                    }
                }
            )
        logging.info(f"{time} - {human.name} {Event.contamination} event")
        logging.debug("{time} - {human.name} was contaminated by {source.name}. "
                      "The probability of getting infected was {p_infection:.6f}"
                      .format(time=time,
                              human=human,
                              source=source,
                              p_infection=p_infection))

    @staticmethod
    def log_recovery(COLLECT_LOGS, human, time, death):
        """
        [summary]

        Args:
            COLLECT_LOGS ([type]): [description]
            human ([type]): [description]
            time ([type]): [description]
            death ([type]): [description]
        """
        if COLLECT_LOGS:
            human.events.append(
                {
                    'human_id': human.name,
                    'event_type': Event.recovered,
                    'time': time,
                    'payload': {
                        'observed':{
                        },
                        'unobserved':{
                            'recovered': not death,
                            'death': death
                        }
                    }
                }
            )
        logging.info(f"{time} - {human.name} {Event.recovered} event")
        logging.debug(f"{time} - {human.name} recovered and is {'' if death else 'not'} {death}.")


    @staticmethod
    def log_static_info(COLLECT_LOGS, city, human, time):
        """
        [summary]

        Args:
            COLLECT_LOGS ([type]): [description]
            city ([type]): [description]
            human ([type]): [description]
            time ([type]): [description]
        """
        h_obs_keys = ['obs_preexisting_conditions',  "obs_age", "obs_sex"]
        h_unobs_keys = ['preexisting_conditions', "age", "sex"]
        obs_payload = {key:getattr(human, key) for key in h_obs_keys}
        unobs_payload = {key:getattr(human, key) for key in h_unobs_keys}
        #
        unobs_payload['is_healthare_worker'] = human.workplace is not None and human.workplace.location_type == "HOSPITAL"
        obs_payload['obs_is_healthcare_worker'] = human.obs_is_healthcare_worker

        if human.does_not_work:
            obs_payload['n_people_workplace'] = 'no people outside my household'
        elif human.workplace.location_type in ['HOSPITAL', 'STORE', 'MISC', 'SENIOR_RESIDENCE']:
            obs_payload['n_people_workplace'] = 'many people'
        elif "WORKPLACE" == human.workplace.location_type:
            obs_payload['n_people_workplace'] = 'few people'
        else:
            obs_payload['n_people_workplace'] = 'no people outside my household'

        obs_payload['household_size'] = len(human.household.residents)

        event = {
                'human_id': human.name,
                'event_type':Event.static_info,
                'time':time,
                'payload':{
                    'observed': obs_payload,
                    'unobserved':unobs_payload
                }
            }
        if COLLECT_LOGS:
            human.events.append(event)
        logging.info(f"{time} - {human.name} {Event.static_info} event")
        logging.debug(f"{time} - {human} static info:\n{event}")
