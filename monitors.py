import json

class StateMonitor(object):
  def __init__(self, f=None):
    self.data = []
    self.f = f or 60

  def run(self, env, city: City):
    while True:
      d = {
          'time': city.time_of_day(),
          'people': len(city.humans),
          'sick': sum([int(h.is_sick) for h in city.humans]),
      }
      self.data.append(d)
      print(city.clock.time_of_day())
      yield env.timeout(self.f/TICK_MINUTE)

  def dump(self, dest:str=None):
    print(json.dumps(self.data, indent=1))



class PlotMonitor(object):
  def __init__(self, f=None):
    self.data = []
    self.f = f or 60

  def run(self, env, city: City):
    fig=plt.figure(figsize=(15, 12))
    while True:
      d = {
          'time': city.clock.time(),
          'htime': city.clock.time_of_day(),
          'sick': sum([int(h.is_sick) for h in city.humans]),
      }
      for k, v in Human.actions.items():
         d[k] = sum(int(h.action == v) for h in city.humans)

      self.data.append(d)
      yield env.timeout(self.f/TICK_MINUTE)
      self.plot()


  def plot(self):
    display.clear_output(wait=True)
    pl.clf()
    time_series = [d['time'] for d in self.data]
    sick_series = [d['sick'] for d in self.data]
    pl.plot(time_series, sick_series, label='sick')
    for k, v in Human.actions.items():
        action_series = [d[k] for d in self.data]
        pl.plot(time_series, action_series, label=k)

    pl.title(f"City at {self.data[-1]['htime']}")
    pl.legend()
    display.display(pl.gcf())

  def dump(self, dest:str=None):
    pass


class LatLonMonitor(object):
  def __init__(self, f=None):
    self.data = []
    self.city_data = {}
    self.f = f or 60

  def run(self, env, city: City):
    self.city_data['parks'] = [
      {'lat': l.lat,
       'lon': l.lon,} for l in city.parks
    ]
    self.city_data['stores'] = [
      {'lat': l.lat ,
       'lon': l.lon,} for l in city.stores
    ]
    fig=plt.figure(figsize=(18, 16))
    while True:
      self.data.extend(
          {'time': city.clock.time_of_day(),
           'is_sick': h.is_sick,
           'lat': h.lat(),
           'lon': h.lon(),
           'human_id': h.name,
           'household_id': h.household.name,
           'location': h.location.name if h.location else None
           } for h in city.humans
      )
      yield env.timeout(self.f/TICK_MINUTE)
      self.plot()


  def plot(self):
    display.clear_output(wait=True)
    pl.clf()
    # PLOT STORES AND PARKS
    lat_series = [d['lat'] for d in self.city_data['parks']]
    lon_series = [d['lon'] for d in self.city_data['parks']]
    s = 250
    pl.scatter(lat_series, lon_series, s=s, marker='o', color='green', label='parks')

    # PLOT STORES AND PARKS
    lat_series = [d['lat'] for d in self.city_data['stores']]
    lon_series = [d['lon'] for d in self.city_data['stores']]
    s = 50
    pl.scatter(lat_series, lon_series, s=s, marker='o', color='black', label='stores')

    lat_series = [d['lat'] for d in self.data]
    lon_series = [d['lon'] for d in self.data]
    c = ['red' if d['is_sick'] else 'blue' for d in self.data]
    s = 5
    pl.scatter(lat_series, lon_series, s=s, marker='^', color=c, label='human')
    sicks = sum([d['is_sick'] for d in self.data])
    pl.title(f"City at {self.data[-1]['time']} - sick:{sicks}")
    pl.legend()
    display.display(pl.gcf())

  def dump(self, dest:str=None):
    pass
