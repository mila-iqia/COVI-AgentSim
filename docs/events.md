Each event is similar to what an individual's app would notice.
So `human_id` is the app owner (`Human`)'s unique id.
`time` is the start of the event.
'payload' is the information about the event.


Following events are reported -
1. `encounter` - When two `Human`s are present at the same `Location` at the same time, this event is recorded for both the `Human`s.

```
{
 "human_id": 999,
 "time": datetime.datetime(2020, 2, 28, 7, 0),
 "event_type": "encounter",
 "payload": {
  "encounter_human_id": 921,
  "duration": 9.0, # seconds
  "distance": 506, # centimeters
  "lat": 181, # x-coordinate
  "lon": 650  # y-coordinate
 }
}
```
2. `symptom_start` - When a symptom shows up an event is recorded. It can be COVID or non-COVID symptoms.

```
{
 "human_id": 999,
 "event_type": "symptom_start",
 "time": "2020-02-28 08:00:00",
 "payload": {
   "covid":True # covid symptom
 }
}
```

3. `contamination` - When a `Human` catches the virus an event is recorded.
```
{
 "human_id": 999,
 "event_type": "contamination",
 "time": "2020-02-28 08:00:00",
 "payload": {}
}
```
4. `test` - When a `Human` is tested an event is recored with it's results.
```
{
 "human_id": 999,
 "event_type": "symptom_start",
 "time": datetime.datetime(2020, 2, 29, 7, 11),
 "payload": {
   "result": True # positive test
 }
}

```
