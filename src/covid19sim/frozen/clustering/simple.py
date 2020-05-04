import dataclasses
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu
import covid19sim.frozen.clustering.base as base


@dataclasses.dataclass
class SimpleCluster:
    """A simple message cluster.

    The default implementation of the 'fit' functions for this base class will
    simply attempt to merge new messages and adjust the risk level of the cluster
    to the average of all messages it aggregates.
    """

    uid: np.uint8
    """Unique Identifier (UID) of the cluster."""

    risk_level: np.uint8
    """Quantified risk level of the cluster."""

    first_update_time: np.int64
    """Cluster creation timestamp (i.e. timestamp of first encounter)."""

    latest_update_time: np.int64
    """Latest cluster update timestamp (i.e. timestamp of latest encounter)."""

    messages: typing.List[mu.EncounterMessage] = dataclasses.field(default_factory=list)
    """List of encounter messages aggregated into this cluster (in added order)."""
    # note: messages above might have been updated from their original state!

    ##########################################
    # private variables (for debugging only!)

    _real_encounter_uids: typing.List[np.uint64] = dataclasses.field(default_factory=list)
    """Real Unique Identifiers (UIDs) of the clustered user(s)."""

    _real_encounter_times: typing.List[np.uint64] = dataclasses.field(default_factory=list)
    """Real timestamp of the clustered encounter(s)."""

    _unclustered_messages: typing.List[mu.GenericMessageType] = dataclasses.field(default_factory=list)
    """List of all messages (encounter+update) messages that were used to update this cluster."""

    def _is_homogenous(self) -> bool:
        """Returns whether this cluster is truly homogenous (i.e. tied to one user) or not."""
        return len(np.unique([m._sender_uid for m in self._unclustered_messages])) <= 1

    @staticmethod
    def create_cluster_from_message(message: mu.GenericMessageType) -> "SimpleCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return SimpleCluster(
            # app-visible stuff below
            uid=message.uid,
            risk_level=message.risk_level
                if isinstance(message, mu.EncounterMessage) else message.new_risk_level,
            first_update_time=message.encounter_time,
            latest_update_time=message.encounter_time
                if isinstance(message, mu.EncounterMessage) else message.update_time,
            messages=[message] if isinstance(message, mu.EncounterMessage)
                else mu.create_encounter_from_update_message(message),
            # debug-only stuff below
            _real_encounter_uids=[message._sender_uid],
            _real_encounter_times=[message._real_encounter_time],
            _unclustered_messages=[message],  # once added, messages here should never be removed
        )

    def fit_encounter_message(self, message: mu.EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        # note: this simplistic implementation will throw if UIDs dont perfectly match;
        # the added message will automatically be used to adjust the cluster's risk level
        self.latest_update_time = max(message.encounter_time, self.latest_update_time)
        self.messages.append(message)  # in this list, encounters may get updated
        self.risk_level = np.uint8(np.round(np.mean([m.risk_level for m in self.messages])))
        self._real_encounter_uids.append(message._sender_uid)
        self._real_encounter_times.append(message._real_encounter_time)
        self._unclustered_messages.append(message)  # in this list, messages NEVER get updated

    def fit_update_message(self, update_message: mu.UpdateMessage):
        """Updates an encounter in the current cluster given a new update message.

        If the update message cannot be applied to any encounter in this cluster, it will be
        returned as-is. Otherwise, if the update message was applied to the cluster, the function
        will return `None`.
        """
        found_match = None
        # TODO: should we assert that all update messages are received in order based on update time?
        #       if the assumption is false, what should we do? currently, we try to apply anyway?
        # TODO: see if check below still valid when update messages are no longer systematically sent
        for encounter_message_idx, encounter_message in enumerate(self.messages):
            if encounter_message.risk_level == update_message.old_risk_level and \
                    encounter_message.uid == update_message.uid and \
                    encounter_message.encounter_time == update_message.encounter_time:
                found_match = (encounter_message_idx, encounter_message)
                break
        if found_match is None:
            return update_message
        self.messages[found_match[0]] = mu.create_updated_encounter_with_message(
            encounter_message=found_match[1], update_message=update_message,
        )
        # note: the 'cluster update time' is still encounter-message-based, not update-message-based
        self.latest_update_time = max(update_message.encounter_time, self.latest_update_time)
        self.risk_level = np.uint8(np.round(np.mean([m.risk_level for m in self.messages])))
        self._real_encounter_uids.append(update_message._sender_uid)
        self._real_encounter_times.append(update_message._real_encounter_time)
        self._unclustered_messages.append(update_message)  # in this list, messages NEVER get updated

    def get_cluster_embedding(self) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        # note: this returns an array of four 'features', i.e. the cluster UID, the cluster's
        #       average encounter risk level, the number of messages in the cluster, and
        #       the first encounter timestamp of the cluster. This array's type will be returned
        #       as np.int64 to insure that no data is lost w.r.t. message counts or timestamps.
        return np.asarray([self.uid, self.risk_level,
                           len(self.messages), self.first_update_time], dtype=np.int64)

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event) for m in self.messages])


class SimplisticClusterManager(base.ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a simplistic clustering strategy where encounters are only combined
    on a timestamp-level basis. This means clusters cannot contain messages with different
    timestamps. The update messages can also never split a cluster into different parts.

    THE UPDATE ALGORITHM IS NON-DETERMINISTIC. Make sure to seed your experiments if you want
    to see reproducible behavior.
    """

    clusters: typing.List[SimpleCluster]
    add_orphan_updates_as_clusters: bool

    def __init__(
            self,
            max_history_ticks_offset: int = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            add_orphan_updates_as_clusters: bool = False,
            rng=np.random,
    ):
        super().__init__(max_history_ticks_offset=max_history_ticks_offset)
        self.add_orphan_updates_as_clusters = add_orphan_updates_as_clusters
        self.rng = rng

    def _add_encounter_message(self, message: mu.EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # simplistic clustering = we are looking for an exact timestamp/uid/risk level match;
        # if one is not found, we create a new cluster
        matched_clusters = []
        for cluster in self.clusters:
            if cluster.uid == message.uid and \
                    cluster.risk_level == message.risk_level and \
                    cluster.first_update_time == message.encounter_time:
                matched_clusters.append(cluster)
        if matched_clusters:
            # the number of matched clusters might be greater than one if update messages caused
            # a cluster signature to drift into another cluster's; we will randomly assign this
            # encounter to one of the two (this is the naive part)
            matched_cluster = self.rng.choice(matched_clusters)
            matched_cluster.fit_encounter_message(message)
        else:
            # create a new cluster for this encounter alone
            self.clusters.append(SimpleCluster.create_cluster_from_message(message))

    def _add_update_message(self, message: mu.UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        matched_clusters = []
        for cluster in self.clusters:
            if cluster.uid == message.uid and cluster.first_update_time == message.encounter_time:
                # found a potential match based on uid and encounter time; check for actual
                # encounters in the cluster with the target risk level to update...
                for encounter in cluster.messages:
                    if encounter.risk_level == message.old_risk_level:
                        matched_clusters.append(cluster)
                        # one matching encounter is sufficient, we can update that cluster
                        break
        if matched_clusters:
            # the number of matched clusters might be greater than one if update messages caused
            # a cluster signature to drift into another cluster's; we will randomly assign this
            # encounter to one of the two (this is the naive part)
            matched_cluster = self.rng.choice(matched_clusters)
            matched_cluster.fit_update_message(message)
        else:
            if self.add_orphan_updates_as_clusters:
                self.clusters.append(SimpleCluster.create_cluster_from_message(message))
            else:
                raise AssertionError(f"could not find any proper cluster match for: {message}")
