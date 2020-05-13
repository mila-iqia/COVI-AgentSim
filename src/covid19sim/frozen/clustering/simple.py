import collections
import dataclasses
import numpy as np
import typing

from covid19sim.frozen.message_utils import EncounterMessage, GenericMessageType, UpdateMessage, \
    TimestampType, RiskLevelType, create_encounter_from_update_message, create_updated_encounter_with_message
from covid19sim.frozen.clustering.base import ClusterIDType, RealUserIDType, TimeOffsetType, \
    ClusterBase, ClusterManagerBase


class SimpleCluster(ClusterBase):
    """A simple encounter message cluster.

    The default implementation of the 'fit' functions for this base class will
    simply attempt to merge new messages and adjust the risk level of the cluster
    to the average of all messages it aggregates.
    """

    messages: typing.List[EncounterMessage] = dataclasses.field(default_factory=list)
    """List of encounter messages aggregated into this cluster (in added order)."""
    # note: messages above might have been updated from their original state!

    skip_homogeneity_checks: bool  # will be toggled on for 'perfect' clustering only

    def __init__(
            self,
            messages: typing.List[EncounterMessage],
            skip_homogeneity_checks: bool = False,
            **kwargs,
    ):
        """Creates a simple cluster, forwarding most args to the base class."""
        super().__init__(
            **kwargs,
        )
        self.messages = messages
        self.skip_homogeneity_checks = skip_homogeneity_checks

    @staticmethod
    def create_cluster_from_message(
            message: GenericMessageType,
            cluster_id: ClusterIDType,
    ) -> "SimpleCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return SimpleCluster(
            # app-visible stuff below
            cluster_id=cluster_id,
            risk_level=message.risk_level
                if isinstance(message, EncounterMessage) else message.new_risk_level,
            first_update_time=message.encounter_time,
            latest_update_time=message.encounter_time,
            messages=[message] if isinstance(message, EncounterMessage)
                else [create_encounter_from_update_message(message)],
            # debug-only stuff below
            _real_encounter_uids=[message._sender_uid],
            _real_encounter_times=[message._real_encounter_time],
            _unclustered_messages=[message],  # once added, messages here should never be removed
        )

    def fit_encounter_message(self, message: EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        if not self.skip_homogeneity_checks:
            assert self.risk_level == message.risk_level
            assert self.latest_update_time == message.encounter_time
            assert self.first_update_time == message.encounter_time
        self.messages.append(message)  # in this list, encounters may get updated
        self._real_encounter_uids.append(message._sender_uid)
        self._real_encounter_times.append(message._real_encounter_time)
        self._unclustered_messages.append(message)  # in this list, messages NEVER get updated

    def fit_update_message(self, update_message: UpdateMessage):
        """Updates an encounter in the current cluster given a new update message.

        If the update message cannot be applied to any encounter in this cluster, it will be
        returned as-is. Otherwise, if the update message was applied to the cluster, the function
        will return `None`.
        """
        if not self.skip_homogeneity_checks:
            assert self.messages[0].uid == update_message.uid
            assert self.latest_update_time == update_message.encounter_time
            assert self.first_update_time == update_message.encounter_time
        found_match_idx = None
        for encounter_message_idx, encounter_message in enumerate(self.messages):
            if encounter_message.risk_level == update_message.old_risk_level and \
                    encounter_message.encounter_time == update_message.encounter_time:
                found_match_idx = encounter_message_idx
                break
        if found_match_idx is None:
            return update_message
        self.messages[found_match_idx] = create_updated_encounter_with_message(
            encounter_message=self.messages[found_match_idx],
            update_message=update_message,
            blind_update=self.skip_homogeneity_checks,
        )
        self._real_encounter_uids.append(update_message._sender_uid)
        self._real_encounter_times.append(update_message._real_encounter_time)
        self._unclustered_messages.append(update_message)  # in this list, messages NEVER get updated

    def get_cluster_embedding(
            self,
            current_timestamp: TimestampType,
            include_cluster_id: bool,
            old_compat_mode: bool = False,
    ) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        actual_risk_level = RiskLevelType(np.round(np.mean([m.risk_level for m in self.messages])))
        if old_compat_mode:
            assert include_cluster_id
            # cid+risk+duraton+day; 4th entry ('day') will be added in the caller
            return np.asarray([self.cluster_id, actual_risk_level, len(self.messages)], dtype=np.int64)
        else:
            # note: this returns an array of four 'features', i.e. the cluster ID, the cluster's
            #       average encounter risk level, the number of messages in the cluster, and
            #       the offset to the first encounter timestamp of the cluster. This array's type
            #       will be returned as np.int64 to insure that no data is lost w.r.t. message
            #       counts or timestamps.
            if include_cluster_id:
                return np.asarray([
                    self.cluster_id, actual_risk_level, len(self.messages),
                    current_timestamp - self.first_update_time  # first/last is the same
                ], dtype=np.int64)
            else:
                return np.asarray([
                    actual_risk_level, len(self.messages),
                    current_timestamp - self.first_update_time  # first/last is the same
                ], dtype=np.int64)

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event) for m in self.messages])

    def get_timestamps(self) -> typing.List[TimestampType]:
        """Returns the list of timestamps for which this cluster possesses at least one encounter."""
        return [self.first_update_time]  # this impl's clusters always only cover a single timestamp

    def get_encounter_count(self) -> int:
        """Returns the number of encounters aggregated inside this cluster."""
        return len(self.messages)


class SimplisticClusterManager(ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a simplistic clustering strategy where encounters are only combined
    on a timestamp-level basis. This means clusters cannot contain messages with different
    timestamps. The update messages can also never split a cluster into different parts.
    """

    clusters: typing.List[SimpleCluster]

    def __init__(
            self,
            max_history_ticks_offset: TimeOffsetType = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
            generate_backw_compat_embeddings: bool = False,
            max_cluster_id: int = 1000,  # let's hope no user ever reaches 1000 simultaneous clusters
    ):
        super().__init__(
            max_history_ticks_offset=max_history_ticks_offset,
            add_orphan_updates_as_clusters=add_orphan_updates_as_clusters,
            generate_embeddings_by_timestamp=generate_embeddings_by_timestamp,
            generate_backw_compat_embeddings=generate_backw_compat_embeddings,
            max_cluster_id=max_cluster_id,
        )

    def _add_encounter_message(self, message: EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # simplistic clustering = we are looking for an exact timestamp/uid/risk level match
        matched_cluster = None
        for cluster in self.clusters:
            if cluster.messages[0].uid == message.uid and \
                    cluster.risk_level == message.risk_level and \
                    cluster.first_update_time == message.encounter_time:
                matched_cluster = cluster
                break
        if matched_cluster is not None:
            matched_cluster.fit_encounter_message(message)
        else:
            new_cluster = SimpleCluster.create_cluster_from_message(message, self.next_cluster_id)
            self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
            self.clusters.append(new_cluster)

    def _add_update_message(self, message: UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        matched_cluster = None
        for cluster in self.clusters:
            # all cluster encounters should have the same uid; just check the first
            if cluster.messages[0].uid == message.uid and \
                    cluster.first_update_time == message.encounter_time:
                # found a potential match based on uid and encounter time; check for actual
                # encounters in the cluster with the target risk level to update...
                for encounter in cluster.messages:
                    assert encounter.uid == message.uid
                    if encounter.risk_level == message.old_risk_level:
                        matched_cluster = cluster
                        # one matching encounter is sufficient, we can update that cluster
                        break
            if matched_cluster:
                break
        if matched_cluster is not None:
            matched_cluster.fit_update_message(message)
        else:
            if self.add_orphan_updates_as_clusters:
                new_cluster = SimpleCluster.create_cluster_from_message(message, self.next_cluster_id)
                self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                self.clusters.append(new_cluster)
            else:
                raise AssertionError(f"could not find any proper cluster match for: {message}")

    def get_embeddings_array(self) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        assert not self.generate_backw_compat_embeddings or self.generate_embeddings_by_timestamp, \
            "original embeddings were generated by timestamp, cannot avoid that for backw compat"
        if self.generate_embeddings_by_timestamp:
            cluster_embeds = collections.defaultdict(list)
            for cluster in self.clusters:
                embed = cluster.get_cluster_embedding(
                    current_timestamp=self.latest_refresh_timestamp,
                    include_cluster_id=True,
                    old_compat_mode=self.generate_backw_compat_embeddings,
                )
                for msg in cluster.messages:
                    cluster_embeds[msg.encounter_time].append([*embed, cluster.latest_update_time])
            flat_output = []
            for timestamp in sorted(cluster_embeds.keys()):
                flat_output.extend(cluster_embeds[timestamp])
            return np.asarray(flat_output)
        else:
            return np.asarray([
                c.get_cluster_embedding(
                    current_timestamp=self.latest_refresh_timestamp,
                    include_cluster_id=False,
                ) for c in self.clusters], dtype=np.int64)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        if self.generate_embeddings_by_timestamp:
            cluster_flags = collections.defaultdict(list)
            for cluster in self.clusters:
                flags = cluster._get_cluster_exposition_flag()
                for msg in cluster.messages:
                    cluster_flags[msg.encounter_time].append(flags)
            flat_output = []
            for timestamp in sorted(cluster_flags.keys()):
                flat_output.extend(cluster_flags[timestamp])
            return np.asarray(flat_output)
        else:
            return np.asarray([c._get_cluster_exposition_flag() for c in self.clusters], dtype=np.uint8)

    def _get_homogeneity_scores(self) -> typing.Dict[RealUserIDType, float]:
        """Returns the homogeneity score for all real users in the clusters.

        The homogeneity score for a user is defined as the number of true encounters involving that
        user divided by the total number of encounters attributed to that user (via clustering). It
        expresses how well the clustering algorithm managed to isolate that user's encounters from
        the encounters of other users. In other words, it expresses how commonly a user was confused
        for other users.

        A homogeneity score of 1 means that the user was only ever assigned to clusters that only
        contained its own encounters. The homogeneity does not reflect how many extra (unnecessary)
        clusters were created by the algorithm.

        Computing this score requires the use of the "real" user IDs, meaning this is only
        possible with simulator data.
        """
        user_true_encounter_counts = collections.defaultdict(int)
        user_total_encounter_count = collections.defaultdict(int)
        for cluster in self.clusters:
            cluster_users = set()
            for msg in cluster.messages:
                user_true_encounter_counts[msg._sender_uid] += 1
                cluster_users.add(msg._sender_uid)
            for user in cluster_users:
                user_total_encounter_count[user] += len(cluster.messages)
        return {user: user_true_encounter_counts[user] / user_total_encounter_count[user]
                for user in user_true_encounter_counts}

    def get_encounters_cluster_mapping(self) -> typing.List[typing.Tuple[EncounterMessage, ClusterIDType]]:
        """Returns a flattened list of encounters mapped to their cluster ids."""
        return [(encounter, c.cluster_id) for c in self.clusters for encounter in c.messages]

    def _get_cluster_count_error(self) -> int:
        """Returns the difference between the number of clusters and the number of unique users.

        Since the number of clusters should correspond to the number of unique users that the
        clustering method can detect, we can compare this value with the actual number of
        unique users that it saw. The absolute difference between the two can inform us on
        how well the clustering method fragmented the encounters.

        Note that an absolute error of 0 does not mean that the clusters were properly matched
        to the right users. It only means that the clustering resulted in the right number of
        users.

        Computing this score requires the use of the "real" user IDs, meaning this is only
        possible with simulator data.
        """
        encountered_users = set()
        for cluster in self.clusters:
            for msg in cluster.messages:
                encountered_users.add(msg._sender_uid)
        return abs(len(encountered_users) - len(self.clusters))
