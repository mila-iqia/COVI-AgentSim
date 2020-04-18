import datetime
from config import INFECTIOUSNESS_ONSET_DAYS
class Event:
    test = 'test'
    encounter = 'encounter'
    symptom_start = 'symptom_start'
    contamination = 'contamination'
    recovered = 'recovered'
    static_info = 'static_info'
    visit = 'visit'
    daily = 'daily'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.symptom_start, Event.contamination, Event.static_info, Event.visit, Event.daily]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, infectee, time):
        h_obs_keys   = ['has_app',
                        'obs_hospitalized', 'obs_in_icu', 'wearing_mask',
                        'obs_lat', 'obs_lon']

        h_unobs_keys = ['carefulness', 'viral_load', 'infectiousness',
                        'symptoms', 'is_exposed', 'is_infectious',
                        'infection_timestamp', 'really_sick',
                        'extremely_sick', 'sex']

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
            u['infectiousness_start_time'] = None if not u['got_exposed'] else human.infection_timestamp + datetime.timedelta(days=human.incubation_days - INFECTIOUSNESS_ONSET_DAYS)
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
                'event_type':Event.encounter,
                'time':time,
                'payload':{'observed':obs_payload, 'unobserved':unobs_payload}
            })


    @staticmethod
    def log_test(human, test_result, test_type, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.test,
                'time': time,
                'payload': {
                    'observed':{
                        'result': test_result,
                        'test_type':test_type
                    },
                    'unobserved':{
                    }

                }
            }
        )

    @staticmethod
    def log_daily(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.daily,
                'time': time,
                'payload': {
                    'observed':{
                    },
                    'unobserved':{
                        'infectiousness': human.infectiousness,
                    }
                }
            }
        )

    @staticmethod
    def log_symptom_start(human, covid, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.symptom_start,
                'time': time,
                'payload': {
                    'observed':{
                        "reported_symptoms": human.all_reported_symptoms
                    },
                    'unobserved':{
                        'covid': covid,
                        "all_symptoms": human.all_symptoms
                    }

                }
            }
        )

    @staticmethod
    def log_exposed(human, source, time):
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
                      'infectiousness_start_time': human.infection_timestamp + datetime.timedelta(days=human.incubation_days - INFECTIOUSNESS_ONSET_DAYS)
                    }

                }
            }
        )

    @staticmethod
    def log_recovery(human, time, death):
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

    @staticmethod
    def log_visit(human, time, location):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.visit,
                'time': time,
                'payload': {
                    'observed':{
                        'location_name': location.name
                    },
                    'unobserved':{
                    }
                }
            }
        )

    @staticmethod
    def log_static_info(city, human, time):
        h_obs_keys = ['obs_preexisting_conditions',  "obs_age", "obs_sex", "obs_is_healthcare_worker"]
        h_unobs_keys = ['preexisting_conditions', "age", "sex", "is_healthcare_worker"]
        obs_payload = {key:getattr(human, key) for key in h_obs_keys}
        unobs_payload = {key:getattr(human, key) for key in h_unobs_keys}

        if human.workplace.location_type in ['healthcare', 'store', 'misc', 'senior_residency']:
            obs_payload['n_people_workplace'] = 'many people'
        elif "workplace" == human.workplace.location_type:
            obs_payload['n_people_workplace'] = 'few people'
        else:
            obs_payload['n_people_workplace'] = 'no people outside my household'

        obs_payload['household_size'] = len(human.household.residents)

        human.events.append(
            {
                'human_id': human.name,
                'event_type':Event.static_info,
                'time':time,
                'payload':{
                    'observed': obs_payload,
                    'unobserved':unobs_payload
                }

            }
        )

class DummyEvent:
    @staticmethod
    def log_encounter(*args, **kwargs):
        pass

    @staticmethod
    def log_test(*args, **kwargs):
        pass

    @staticmethod
    def log_symptom_start(*args, **kwargs):
        pass

    @staticmethod
    def log_recovery(*args, **kwargs):
        pass

    @staticmethod
    def log_exposed(*args, **kwargs):
        pass

    @staticmethod
    def log_static_info(*args, **kwargs):
        pass

    @staticmethod
    def log_visit(*args, **kwargs):
        pass
