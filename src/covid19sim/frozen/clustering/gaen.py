import collections
import datetime
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu
from covid19sim.frozen.clustering.base import ClusterIDType, TimestampType, \
    TimeOffsetType, ClusterBase, ClusterManagerBase, MessagesArrayType, RealUserIDType, \
    UpdateMessageBatchType


class GAENCluster(ClusterBase):
    """A blind and GAEN-compatible encounter message cluster.

    The default implementation of the 'fit' functions for this base class will attempt
    to merge new messages if they share a risk level and split the cluster if partial
    updates are received.

    This cluster may also contain encounters for different timestamps.
    """

    messages_by_timestamp: typing.Dict[TimestampType, typing.List[mu.EncounterMessage]]
    """Timestamp-to-encounter map of all messages owned by this cluster."""

    def __init__(
            self,
            messages: typing.List[mu.EncounterMessage],
            **kwargs,
    ):
        """Creates a cluster, forwarding most args to the base class."""
        super().__init__(**kwargs)
        self.messages_by_timestamp = {}
        self._reset_messages_by_timestamps(messages)

    def _reset_messages_by_timestamps(self, messages: typing.List[mu.EncounterMessage]):
        """Resets the internal messages-by-timestamps mapping with a new list of messages."""
        self.messages_by_timestamp = {}
        for m in messages:
            if m.encounter_time not in self.messages_by_timestamp:
                self.messages_by_timestamp[m.encounter_time] = []
            self.messages_by_timestamp[m.encounter_time].append(m)

    @staticmethod
    def create_cluster_from_message(
            message: mu.GenericMessageType,
            cluster_id: ClusterIDType,
    ) -> "GAENCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return GAENCluster(
            # app-visible stuff below
            cluster_id=cluster_id,
            risk_level=message.risk_level
                if isinstance(message, mu.EncounterMessage) else message.new_risk_level,
            first_update_time=message.encounter_time,
            latest_update_time=message.encounter_time,
            messages=[message] if isinstance(message, mu.EncounterMessage)
                else [mu.create_encounter_from_update_message(message)],
            # debug-only stuff below
            _real_encounter_uids={message._sender_uid},
            _real_encounter_times={message._real_encounter_time},
        )

    def _force_fit_encounter_message(
            self,
            message: mu.EncounterMessage,
    ):
        """Updates the current cluster given a new encounter message."""
        # update the cluster time with the new message's encounter time (if more recent)
        self.latest_update_time = max(message.encounter_time, self.latest_update_time)
        if message.encounter_time not in self.messages_by_timestamp:
            self.messages_by_timestamp[message.encounter_time] = []
        else:
            assert self.messages_by_timestamp[message.encounter_time][0].risk_level == message.risk_level
            assert self.messages_by_timestamp[message.encounter_time][0].encounter_time == message.encounter_time
        self.messages_by_timestamp[message.encounter_time].append(message)
        self._real_encounter_uids.add(message._sender_uid)
        self._real_encounter_times.add(message._real_encounter_time)

    def _force_fit_encounter_message_batch(
            self,
            messages: typing.List[mu.EncounterMessage],
    ):
        """Updates the current cluster given a batch of new encounter messages."""
        # NOTE: code below assumes all encounter messages in the batch have the same risk level
        if not messages:
            return
        # update the cluster time with the new message's encounter time (if more recent)
        self.latest_update_time = max(messages[0].encounter_time, self.latest_update_time)
        if messages[0].encounter_time not in self.messages_by_timestamp:
            self.messages_by_timestamp[messages[0].encounter_time] = []
        else:
            assert self.messages_by_timestamp[messages[0].encounter_time][0].risk_level == messages[0].risk_level
            assert self.messages_by_timestamp[messages[0].encounter_time][0].encounter_time == messages[0].encounter_time
        self.messages_by_timestamp[messages[0].encounter_time].extend(messages)
        self._real_encounter_uids.update([m._sender_uid for m in messages])
        self._real_encounter_times.update([m._real_encounter_time for m in messages])

    def fit_encounter_message(
            self,
            message: mu.EncounterMessage,
    ) -> typing.Optional[mu.EncounterMessage]:
        """Updates the current cluster given a new encounter message."""
        assert message.risk_level == self.risk_level, "cluster and new encounter message risks mismatch"
        return self._force_fit_encounter_message(message=message)

    def fit_update_message(
            self,
            update_message: mu.UpdateMessage,
    ) -> typing.Optional[typing.Union[mu.UpdateMessage, "GAENCluster"]]:
        """Updates an encounter in the current cluster given a new update message.

        If this cluster gets split as a result of the update, the function will return the newly
        created cluster. Otherwise, if the update message cannot be applied to any encounter in this
        cluster, it will be returned as-is. Finally, if the update message was applied to the cluster
        without splitting it, the function will return `None`.
        """
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_message.old_risk_level == self.risk_level, "cluster & update message old risk mismatch"
        # quick-exit if this cluster does not contain the timestamp for the encounter
        if update_message.encounter_time not in self.messages_by_timestamp:
            # could not find any match for the update message; send it back to the manager
            return update_message
        assert update_message.old_risk_level == \
               self.messages_by_timestamp[update_message.encounter_time][0].risk_level
        assert update_message.encounter_time == \
               self.messages_by_timestamp[update_message.encounter_time][0].encounter_time
        # quick exit if we cannot find a GAEN key match at the right timestamp
        found_match_idx = None
        for idx, m in enumerate(self.messages_by_timestamp[update_message.encounter_time]):
            if update_message.uid == m.uid:
                found_match_idx = idx
                break
        if found_match_idx is None:
            return update_message
        if len(self.messages_by_timestamp) == 1 and \
                len(self.messages_by_timestamp[update_message.encounter_time]) == 1:
            # we can self-update without splitting; do that
            old_encounter = self.messages_by_timestamp[update_message.encounter_time][0]
            new_encounter = mu.create_updated_encounter_with_message(
                encounter_message=old_encounter, update_message=update_message,
            )
            self.messages_by_timestamp = {new_encounter.encounter_time: [new_encounter]}
            self.risk_level = new_encounter.risk_level
            self._real_encounter_uids.add(update_message._sender_uid)
            self._real_encounter_times.add(update_message._real_encounter_time)
            return None
        else:
            # we have multiple messages in this cluster, and the update can only apply to one;
            # ... we need to split the cluster into two, where only the new one will be updated
            message_to_transfer = self.messages_by_timestamp[update_message.encounter_time].pop(found_match_idx)
            if not self.messages_by_timestamp[update_message.encounter_time]:
                del self.messages_by_timestamp[update_message.encounter_time]
            return self.create_cluster_from_message(mu.create_updated_encounter_with_message(
                encounter_message=message_to_transfer, update_message=update_message,
            ), cluster_id=None)  # cluster id will be assigned properly in manager
            # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars

    def fit_update_message_batch(
            self,
            update_messages: typing.Dict[TimestampType, typing.List[mu.UpdateMessage]],
    ) -> typing.Tuple[typing.Dict[TimestampType, typing.List[mu.UpdateMessage]], typing.Optional["GAENCluster"]]:
        """Updates encounters in the current cluster given a list of new update messages.

        If this cluster gets split as a result of the update, the function will return the newly
        created cluster. Otherwise, if the update messages cannot be applied to any encounter in this
        cluster, they will be returned as-is. Finally, if any update message was applied to the cluster
        without splitting it, the function will return the remaining updates.
        """
        if not update_messages:
            return {}, None
        # NOTE: code below assumes all update messages in the batch have the same risk level
        matching_timestamps = set(update_messages.keys()) & set(self.messages_by_timestamp.keys())
        # quick-exit if this cluster does not contain any of the encounter timestamps
        if not matching_timestamps:
            return update_messages, None
        found_matches = collections.defaultdict(dict)  # timestamp-to-internal-message-index-to-popped-update-message
        for timestamp in matching_timestamps:
            internal_uids = [m.uid for m in self.messages_by_timestamp[timestamp]]
            for idx, msg in reversed(list(enumerate(update_messages[timestamp]))):
                if msg.uid in internal_uids:
                    found_matches[timestamp][internal_uids.index(msg.uid)] = \
                        update_messages[timestamp].pop(idx)
                    if not update_messages[timestamp]:
                        del update_messages[timestamp]
        if not found_matches:
            return update_messages, None
        if len(self.messages_by_timestamp) == len(found_matches):
            if all([
                len(found_matches[matched_timestamp]) == len(self.messages_by_timestamp[matched_timestamp])
                for matched_timestamp in self.messages_by_timestamp.keys()
            ]):
                new_risk_level = None
                # we can apply simultaneous updates to all messages in this cluster and avoid splitting; do that
                for timestamp in self.messages_by_timestamp.keys():
                    old_encounters = self.messages_by_timestamp[timestamp]
                    new_encounters = []
                    for encounter_idx, old_encounter in enumerate(old_encounters):
                        update_message = found_matches[timestamp][encounter_idx]
                        new_encounters.append(mu.create_updated_encounter_with_message(
                            encounter_message=old_encounter, update_message=update_message,
                        ))
                    self._real_encounter_uids.update([m._sender_uid for m in found_matches[timestamp].values()])
                    self._real_encounter_times.update([m._real_encounter_time for m in found_matches[timestamp].values()])
                    self.messages_by_timestamp[timestamp] = new_encounters
                    if new_risk_level is None:
                        new_risk_level = new_encounters[0].risk_level
                    else:
                        assert new_risk_level == new_encounters[0].risk_level
                self.risk_level = new_risk_level
                return update_messages, None
        # we lack a bunch of update messages, so we still need to split
        messages_to_transfer = [
            mu.create_updated_encounter_with_message(
                encounter_message=self.messages_by_timestamp[timestamp][match_idx],
                update_message=update_message,
            )
            for timestamp, matches in found_matches.items()
            for match_idx, update_message in matches.items()
        ]
        self._reset_messages_by_timestamps([
            self.messages_by_timestamp[timestamp][encounter_idx]
            for timestamp, encounters in self.messages_by_timestamp.items()
            for encounter_idx in reversed(range(len(encounters)))
            if timestamp not in found_matches or encounter_idx not in found_matches[timestamp]
        ])
        # todo: create cluster from message batch
        new_cluster = self.create_cluster_from_message(
            messages_to_transfer[0],
            cluster_id=None,  # cluster id will be assigned properly in manager
        )
        if len(messages_to_transfer) > 1:
            new_cluster._force_fit_encounter_message_batch(messages_to_transfer[1:])
        return update_messages, new_cluster

    def fit_cluster(
            self,
            cluster: "GAENCluster",
    ) -> None:
        """Updates this cluster to incorporate all the encounters in the provided cluster.

        This function will throw if anything funky is detected.

        WARNING: the cluster provided to this function must be discarded after this call!
        If this is not done, we will have duplicated messages somewhere in the manager...
        """
        # @@@@ TODO: batch-fit clusters? (will avoid multi loop+extend below)
        assert self.risk_level == cluster.risk_level
        self.first_update_time = min(self.first_update_time, cluster.first_update_time)
        self.latest_update_time = max(self.latest_update_time, cluster.latest_update_time)
        # note: encounters should NEVER be duplicated! if these get copied here, we expect
        #       that the provided 'cluster' object will get deleted!
        for timestamp, encounters in cluster.messages_by_timestamp.items():
            if not cluster.messages_by_timestamp:
                continue
            if timestamp not in self.messages_by_timestamp:
                self.messages_by_timestamp[timestamp] = []
            self.messages_by_timestamp[timestamp].extend(encounters)
            # FIXME: there are messages getting duplicated somewhere, this is pretty bad @@@@@
            # assert len(self.messages_by_timestamp[timestamp]) == \
            #     len(np.unique([m.uid for m in self.messages_by_timestamp[timestamp]])), \
            #     "found uid collision while merging cluster; go buy a lottery ticket?"
        # we can make sure whoever tries to use the other cluster again will have a bad surprise...
        cluster.messages_by_timestamp = None
        self._real_encounter_uids.update(cluster._real_encounter_uids)
        self._real_encounter_times.update(cluster._real_encounter_times)

    def get_cluster_embedding(
            self,
            current_timestamp: TimestampType,
            include_cluster_id: bool,
            old_compat_mode: bool = False,
    ) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        if old_compat_mode:
            assert include_cluster_id
            # we want 4 values: cluster id + cluster risk + encounter count + time offset
            # ... the last value (time offset) will be added in the manager who calls this
            nb_encounters_on_target_day = 0
            for timestamp, messages in self.messages_by_timestamp.items():
                if timestamp.date() == current_timestamp.date():
                    nb_encounters_on_target_day += len(messages)
            if nb_encounters_on_target_day:
                return np.asarray([
                    self.cluster_id,
                    self.risk_level,
                    nb_encounters_on_target_day,
                ], dtype=np.int64)
            else:
                return None  # the cluster does not have encounters on that day, skip it
        else:
            raise NotImplementedError

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event)
                    for messages in self.messages_by_timestamp.values()
                    for m in messages])

    def get_timestamps(self) -> typing.List[TimestampType]:
        """Returns the list of timestamps for which this cluster possesses at least one encounter."""
        return list(self.messages_by_timestamp.keys())

    def get_encounter_count(self) -> int:
        """Returns the number of encounters aggregated inside this cluster."""
        return sum([len(msgs) for msgs in self.messages_by_timestamp.values()])

    def get_encounter_uids(self) -> typing.List[mu.UIDType]:
        """Returns the list of all encounter GAEN keys (or uids) aggregated into this cluster."""
        uids = [msg.uid for msgs in self.messages_by_timestamp.values() for msg in msgs]
        assert len(np.unique(uids)) == len(uids), "found collision in a cluster"
        return uids


class GAENClusterManager(ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a GAEN-compatible clustering strategy where encounters can be combined
    across timestamps as long as their risk levels are the same. Update messages can split clusters
    into two parts, where only one part will receive an update. Merging of identical clusters will
    happen periodically to keep the overall count low.
    """

    clusters: typing.List[GAENCluster]

    def __init__(
            self,
            max_history_offset: TimeOffsetType,
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
            generate_backw_compat_embeddings: bool = False,
            max_cluster_id: int = 1000,  # let's hope no user ever reaches 1000 simultaneous clusters
    ):
        super().__init__(
            max_history_offset=max_history_offset,
            add_orphan_updates_as_clusters=add_orphan_updates_as_clusters,
            generate_embeddings_by_timestamp=generate_embeddings_by_timestamp,
            generate_backw_compat_embeddings=generate_backw_compat_embeddings,
            max_cluster_id=max_cluster_id,
        )

    def _merge_clusters(self):
        """Merges clusters that have the exact same signature (because of updates)."""
        cluster_matches, reserved_idxs_for_merge = [], []
        for base_cluster_idx, cluster in enumerate(self.clusters):
            matched_cluster_idxs = []
            for target_cluster_idx in reversed(range(base_cluster_idx + 1, len(self.clusters))):
                if target_cluster_idx in reserved_idxs_for_merge:
                    continue
                if cluster.risk_level == self.clusters[target_cluster_idx].risk_level:
                    matched_cluster_idxs.append(target_cluster_idx)
            cluster_matches.append(matched_cluster_idxs)
            reserved_idxs_for_merge.extend(matched_cluster_idxs)
        to_keep = []
        for base_cluster_idx, (cluster, target_idxs) in enumerate(zip(self.clusters, cluster_matches)):
            if base_cluster_idx in reserved_idxs_for_merge:
                continue
            for target_idx in target_idxs:
                target_cluster = self.clusters[target_idx]
                cluster.fit_cluster(target_cluster)
            to_keep.append(cluster)
        self.clusters = to_keep

    def cleanup_clusters(self, current_timestamp: TimestampType):
        """Gets rid of clusters that are too old given the current timestamp, and single encounters
        inside clusters that are too old as well."""
        to_keep = []
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_update_offset = current_timestamp - cluster.latest_update_time
            if cluster_update_offset < self.max_history_offset:
                for batch_timestamp in list(cluster.messages_by_timestamp.keys()):
                    message_update_offset = current_timestamp - batch_timestamp
                    if message_update_offset >= self.max_history_offset:
                        del cluster.messages_by_timestamp[batch_timestamp]
                if cluster.messages_by_timestamp:
                    to_keep.append(cluster)
        self.clusters = to_keep

    def add_messages(
            self,
            messages: MessagesArrayType,
            cleanup: bool = True,
            current_timestamp: typing.Optional[TimestampType] = None,  # will use internal latest if None
    ):
        """Dispatches the provided messages to the correct internal 'add' function based on type."""
        super().add_messages(messages=messages, cleanup=False)
        self._merge_clusters()
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_encounter_message(self, message: mu.EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        for cluster in self.clusters:
            if cluster.risk_level == message.risk_level:
                cluster._force_fit_encounter_message(message)
                return
        self._add_new_cluster_from_message(message)

    def _add_encounter_message_batch(self, messages: typing.List[mu.EncounterMessage], cleanup: bool = True):
        """Fits a batch of encounter messages to existing clusters, and forward the remaining to non-batch impl."""
        if not messages:
            return
        # we assume all encounter messages in the batch have the same risk level, & are not outdated
        for cluster in self.clusters:
            if cluster.risk_level == messages[0].risk_level:
                cluster._force_fit_encounter_message_batch(messages)
                return
        new_cluster = GAENCluster.create_cluster_from_message(messages[0], self.next_cluster_id)
        new_cluster._force_fit_encounter_message_batch(messages[1:])
        self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
        self.clusters.append(new_cluster)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_update_message(self, message: mu.UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # update-message-to-encounter-message-matching should not be uncertain; we will
        # go through all clusters and fit the update message to the first instance that will take it
        found_adopter = False
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster.risk_level != message.old_risk_level:
                # gaen clusters should always reflect the risk level of all their encounters; if
                # we can't match the risk level here, there's no way the update can apply to it
                continue
            fit_result = cluster.fit_update_message(message)
            if fit_result is None or isinstance(fit_result, GAENCluster):
                if fit_result is not None and isinstance(fit_result, GAENCluster):
                    fit_result.cluster_id = self.next_cluster_id
                    self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                    # to keep the results identical with/without batching, insert at curr index + 1
                    self.clusters.insert(cluster_idx + 1, fit_result)
                found_adopter = True
                break
        if not found_adopter and self.add_orphan_updates_as_clusters:
            self._add_new_cluster_from_message(message)
        elif not found_adopter:
            raise AssertionError(f"could not find adopter for: {message}")

    def _add_update_message_batch(self, messages: UpdateMessageBatchType, cleanup: bool = True):
        """Fits a batch of update messages to existing clusters, and forwards the remaining to non-batch impl."""
        # we assume all update messages in the batch have the same old/new risk levels, & are not outdated
        assert isinstance(messages, dict) and messages, "missing implementation for non-timestamped batches"
        batch_risk_level = messages[next(iter(messages.keys()))][0].old_risk_level
        cluster_idx = 0
        while cluster_idx < len(self.clusters):
            cluster = self.clusters[cluster_idx]
            if cluster.risk_level != batch_risk_level:
                # gaen clusters should always reflect the risk level of all their encounters; if
                # we can't match the risk level here, there's no way the update can apply to it
                cluster_idx += 1
                continue
            messages, new_cluster = cluster.fit_update_message_batch(messages)
            if new_cluster is not None:
                new_cluster.cluster_id = self.next_cluster_id
                self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                # to keep the results identical with/without batching, insert at curr index + 1
                self.clusters.insert(cluster_idx + 1, new_cluster)
                cluster_idx += 1  # skip that cluster if there are still updates to apply
            if not any([len(msgs) for msgs in messages.values()]):
                break  # all messages got adopted
            cluster_idx += 1
        if messages and self.add_orphan_updates_as_clusters:
            self._add_new_cluster_from_message_batch(messages)
        elif messages:
            raise AssertionError(f"could not find adopters for {len(messages)} updates")
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_new_cluster_from_message(self, message: mu.GenericMessageType):
        """Creates and adds a new cluster in the internal structs while cycling the cluster ids."""
        new_cluster = GAENCluster.create_cluster_from_message(message, self.next_cluster_id)
        self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
        self.clusters.append(new_cluster)

    def _add_new_cluster_from_message_batch(self, messages: MessagesArrayType):
        """Creates and adds a new cluster in the internal structs while cycling the cluster ids."""
        assert isinstance(messages, dict) and messages, "missing implementation for non-timestamped batches"
        flat_messages = [m for msgs in messages.values() for m in msgs]
        if not flat_messages:
            return
        assert self.add_orphan_updates_as_clusters
        first_update_message = flat_messages.pop(0)
        assert isinstance(first_update_message, mu.UpdateMessage)
        new_cluster = GAENCluster.create_cluster_from_message(first_update_message, self.next_cluster_id)
        new_encounters = [mu.create_encounter_from_update_message(m) for m in flat_messages]
        new_cluster._force_fit_encounter_message_batch(new_encounters)
        self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
        self.clusters.append(new_cluster)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        if not self.generate_backw_compat_embeddings or not self.generate_embeddings_by_timestamp:
            raise NotImplementedError  # must keep 1:1 mapping with embedding!
        # assume the latest refresh timestamp is up-to-date FIXME should we pass in curr timestamp as above?
        output = []
        target_timestamp = self.latest_refresh_timestamp - self.max_history_offset
        while target_timestamp <= self.latest_refresh_timestamp:
            for cluster in self.clusters:
                cluster_timestamp_match = False
                cluster_contains_matching_exposition = False
                for cluster_timestamp, messages in cluster.messages_by_timestamp.items():
                    if cluster_timestamp.date() == target_timestamp.date():
                        cluster_timestamp_match = True
                        cluster_contains_matching_exposition |= \
                            any([m._exposition_event for m in messages])
                        if cluster_contains_matching_exposition:
                            break
                if cluster_timestamp_match:
                    output.append(cluster_contains_matching_exposition)
            target_timestamp += datetime.timedelta(days=1)
        return np.asarray(output)

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
            tot_message_cout = 0
            for _, msgs in cluster.messages_by_timestamp.items():
                for msg in msgs:
                    user_true_encounter_counts[msg._sender_uid] += 1
                    cluster_users.add(msg._sender_uid)
                    tot_message_cout += 1
            for user in cluster_users:
                user_total_encounter_count[user] += tot_message_cout
        return {user: user_true_encounter_counts[user] / user_total_encounter_count[user]
                for user in user_true_encounter_counts}

    def get_encounters_cluster_mapping(self) -> typing.List[typing.Tuple[mu.EncounterMessage, ClusterIDType]]:
        """Returns a flattened list of encounters mapped to their cluster ids."""
        return [
            (encounter, c.cluster_id)
            for c in self.clusters
            for encounters in c.messages_by_timestamp.values()
            for encounter in encounters
        ]

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
            for msgs in cluster.messages_by_timestamp.values():
                for msg in msgs:
                    encountered_users.add(msg._sender_uid)
        return abs(len(encountered_users) - len(self.clusters))
