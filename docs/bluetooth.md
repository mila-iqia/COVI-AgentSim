## Bluetooth Message Passing (NHS/CTN)

We want to exchange encounter messages between phones for each "high-risk" encounter. A high-risk encounter is one which
takes more than 5 minutes at a distance under 2 meters. Smartphones cannot directly measure distance, but they can measure bluetooth signal strength. 
Bluetooth signal strength is measured in RSSI which is log-scale decibel \cite{https://www.netspotapp.com/what-is-rssi-level.html}.

There is an approximate way to translate RSSI into distance, with a formula that looks like: Distance = 10 ^ ((Measured Power – RSSI)/(10 * N))
To approximately model the noise in the simulator, and because we know the ground-truth distance, we propose to simply take that ground truth 
distance and add noise to it to change whether or not a contact is found. 

We think that there are two kinds of  also add another noise to model, first on which is based on the human who had the contact (i.e., phone-manufacturer bluetooth signal noise) 
which is added to each of their potential contacts. Second would be a per-contact noise based on environmental factors like walls.
To start off, let's just focus on the first noise because modelling the second noise implies a bias in the liklihood of infectious. I.e.,
it takes care in-part of the problem of "people through the wall sent me a message but couldn't infect me".

so what we'll implement before we get real-world data is noise 1: a per-user truncated norm noise that is proportional to the distance, i.e. (1 meter gt + 1 meter * 0%-100% error centered at 50%) 
