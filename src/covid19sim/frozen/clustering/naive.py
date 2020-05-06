import collections
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu
from covid19sim.frozen.clustering.base import ClusterIDType, TimestampType, \
    TimeOffsetType, ClusterBase, ClusterManagerBase, MessagesArrayType


def check_uids_match(
        map1: typing.Dict[TimestampType, typing.List[mu.EncounterMessage]],
        map2: typing.Dict[TimestampType, typing.List[mu.EncounterMessage]],
) -> bool:
    """Returns whether the overlaps in the provided timestamp-to-uid dicts are compatible.

    Note: if there are no overlapping UIDs in the two sets, the function will return `False`.
    """
    overlapping_timestamps = list(set(map1.keys()) & set(map2.keys()))
    return overlapping_timestamps and \
        all([map1[t][0].uid == map2[t][0].uid for t in overlapping_timestamps])


class NaiveCluster(ClusterBase):
    """A naive message cluster.

    The default implementation of the 'fit' functions for this base class will
    attempt to merge new messages across uid rolls and create new clusters if the risk
    level of the cluster is not uniform for all messages it aggregates.
    """

    messages_by_timestamp: typing.Dict[TimestampType, typing.List[mu.EncounterMessage]]
    """Timestamp-to-encounter map of all messages owned by this cluster."""

    def __init__(
            self,
            messages: typing.List[mu.EncounterMessage],
            **kwargs,
    ):
        """Creates a naive cluster, forwarding most args to the base class."""
        super().__init__(**kwargs)
        self.messages_by_timestamp = {m.encounter_time: [m] for m in messages}

    @staticmethod
    def create_cluster_from_message(
            message: mu.GenericMessageType,
            cluster_id: typing.Optional[ClusterIDType] = None,  # used in the output embeddings
    ) -> "NaiveCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return NaiveCluster(
            # app-visible stuff below
            cluster_id=cluster_id,
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

    def _get_encounter_match_score(
            self,
            message: mu.EncounterMessage,
            ticks_per_uid_roll: TimeOffsetType = 24 * 60 * 60,  # one tick per second, one roll per day
    ) -> int:
        """Returns the match score between the provided encounter message and this cluster.

        A negative return value means the message & cluster cannot correspond to the same person. A
        zero value means that no match can be established with certainty (e.g. because the message
        or cluster is too old). A positive value indicates a (possibly partial) match. A high
        positive value means the match is more likely. If the returned value is equal to
        `message_uid_bit_count`, the match is perfect.

        Returns:
            The best match score (an integer in `[-1,message_uid_bit_count]`).
        """
        # IMPORTANT NOTE: >10% OF THE SIMULATION TIME IS SPENT HERE --- KEEP IT LIGHT
        if message.risk_level != self.risk_level:
            # if the cluster's risk level differs from the given encounter's risk level, there is
            # no way to match the two, as the cluster should be updated first to that risk level,
            # or we are looking at different users entirely
            return -1
        if message.encounter_time in self.messages_by_timestamp and \
                self.messages_by_timestamp[message.encounter_time][0].uid != message.uid:
            # if we already have a stored uid at the new encounter's timestamp with a different
            # uid, there's no way this message can be merged into this cluster
            return -1
        t_range = range(
            message.encounter_time - TimeOffsetType(mu.message_uid_bit_count) + 1,
            message.encounter_time + 1
        )
        for timestamp in reversed(t_range):
            if timestamp in self.messages_by_timestamp:
                # we can pick one encounter from the list, they should all match the same way...?
                old_encounter = self.messages_by_timestamp[timestamp][0]
                match_score = mu.find_encounter_match_score(
                    # TODO: what happens if the received message is actually late, and we have an
                    #       'old message' that is more recent? (match scorer will throw)
                    old_encounter,
                    message,
                    ticks_per_uid_roll,
                )
                if match_score > -1:
                    # since the search is reversed, we will always start with the best potential
                    # matches and go down in terms of score; returning first hit should be OK
                    return match_score
        return -1  # did not find jack

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
            assert self.messages_by_timestamp[message.encounter_time][0].uid == message.uid
            assert self.messages_by_timestamp[message.encounter_time][0].encounter_time == message.encounter_time
        self.messages_by_timestamp[message.encounter_time].append(message)
        self._real_encounter_uids.append(message._sender_uid)
        self._real_encounter_times.append(message._real_encounter_time)
        self._unclustered_messages.append(message)  # in this list, messages NEVER get updated

    def _force_fit_encounter_message_batch(
            self,
            messages: typing.List[mu.EncounterMessage],
    ):
        """Updates the current cluster given a batch of new encounter messages."""
        if not messages:
            return
        # update the cluster time with the new message's encounter time (if more recent)
        self.latest_update_time = max(messages[0].encounter_time, self.latest_update_time)
        if messages[0].encounter_time not in self.messages_by_timestamp:
            self.messages_by_timestamp[messages[0].encounter_time] = []
        else:
            assert self.messages_by_timestamp[messages[0].encounter_time][0].uid == messages[0].uid
            assert self.messages_by_timestamp[messages[0].encounter_time][0].encounter_time == messages[0].encounter_time
        self.messages_by_timestamp[messages[0].encounter_time].extend(messages)
        self._real_encounter_uids.extend([m._sender_uid for m in messages])
        self._real_encounter_times.extend([m._real_encounter_time for m in messages])
        self._unclustered_messages.extend(messages)

    def fit_encounter_message(
            self,
            message: mu.EncounterMessage,
            ticks_per_uid_roll: TimeOffsetType = 24 * 60 * 60,  # one tick per second, one roll per day
            minimum_match_score: int = 1,  # means we should at least find a 1-bit uid match
    ) -> typing.Optional[mu.EncounterMessage]:
        """Updates the current cluster given a new encounter message.

        If the already-clustered encounter messages cannot be at least partially matched with the
        new encounter, the given message will be returned as-is. Otherwise, it will be added to
        the cluster, and the function will return `None`.
        """
        assert message.risk_level == self.risk_level, "cluster and new encounter message risks mismatch"
        best_match_score = self._get_encounter_match_score(
            message=message,
            ticks_per_uid_roll=ticks_per_uid_roll,
        )
        if best_match_score < minimum_match_score:
            # ask the manager to add the message as a new cluster instead of merging it in
            return message
        return self._force_fit_encounter_message(message=message)

    def fit_update_message(
            self,
            update_message: mu.UpdateMessage,
    ) -> typing.Optional[typing.Union[mu.UpdateMessage, "NaiveCluster"]]:
        """Updates an encounter in the current cluster given a new update message.

        If this cluster gets split as a result of the update, the function will return the newly
        created cluster. Otherwise, if the update message cannot be applied to any encounter in this
        cluster, it will be returned as-is. Finally, if the update message was applied to the cluster
        without splitting it, the function will return `None`.
        """
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_message.old_risk_level == self.risk_level, "cluster & update message old risk mismatch"
        # quick-exit: if this cluster does not contain the timestamp for the encounter, or if the
        # cluster contains a different uid for that timestamp, there is no way this message is compatible
        if update_message.encounter_time not in self.messages_by_timestamp or \
                self.messages_by_timestamp[update_message.encounter_time][0].uid != update_message.uid:
            # could not find any match for the update message; send it back to the manager
            return update_message
        assert update_message.old_risk_level == \
               self.messages_by_timestamp[update_message.encounter_time][0].risk_level
        assert update_message.uid == \
               self.messages_by_timestamp[update_message.encounter_time][0].uid
        assert update_message.encounter_time ==\
               self.messages_by_timestamp[update_message.encounter_time][0].encounter_time
        if len(self.messages_by_timestamp) == 1 and \
                len(self.messages_by_timestamp[update_message.encounter_time]) == 1:
            # we can self-update without splitting; do that
            old_encounter = self.messages_by_timestamp[update_message.encounter_time][0]
            new_encounter = mu.create_updated_encounter_with_message(
                encounter_message=old_encounter, update_message=update_message,
            )
            self.messages_by_timestamp = {new_encounter.encounter_time: [new_encounter]}
            self.risk_level = new_encounter.risk_level
            self._real_encounter_uids.append(update_message._sender_uid)
            self._real_encounter_times.append(update_message._real_encounter_time)
            self._unclustered_messages.append(update_message)  # in this list, messages NEVER get updated
            return None
        else:
            # we have multiple messages in this cluster, and the update can only apply to one;
            # ... we need to split the cluster into two, where only the new one will be updated
            message_to_transfer = self.messages_by_timestamp[update_message.encounter_time].pop(0)
            if not self.messages_by_timestamp[update_message.encounter_time]:
                del self.messages_by_timestamp[update_message.encounter_time]
            return self.create_cluster_from_message(mu.create_updated_encounter_with_message(
                encounter_message=message_to_transfer, update_message=update_message,
            ))
            # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars

    def fit_update_message_batch(
            self,
            update_messages: typing.List[mu.UpdateMessage],
    ) -> typing.Tuple[typing.List[mu.UpdateMessage], typing.Optional["NaiveCluster"]]:
        if not update_messages:
            return [], None
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_messages[0].old_risk_level == self.risk_level
        # quick-exit: if this cluster does not contain the timestamp for the encounter, or if the
        # cluster contains a different uid for that timestamp, there is no way this message is compatible
        if update_messages[0].encounter_time not in self.messages_by_timestamp or \
                self.messages_by_timestamp[update_messages[0].encounter_time][0].uid != update_messages[0].uid:
            # could not find any match for the update message; send it back to the manager
            return update_messages, None
        nb_matches = min(len(update_messages),
                         len(self.messages_by_timestamp[update_messages[0].encounter_time]))
        if len(self.messages_by_timestamp) == 1 and \
                len(self.messages_by_timestamp[update_messages[0].encounter_time]) == 1:
            # we can trivially self-update without splitting; do that
            old_encounter = self.messages_by_timestamp[update_messages[0].encounter_time][0]
            new_encounter = mu.create_updated_encounter_with_message(
                encounter_message=old_encounter, update_message=update_messages[0],
            )
            self.messages_by_timestamp = {new_encounter.encounter_time: [new_encounter]}
            self.risk_level = new_encounter.risk_level
            self._real_encounter_uids.append(update_messages[0]._sender_uid)
            self._real_encounter_times.append(update_messages[0]._real_encounter_time)
            self._unclustered_messages.append(update_messages[0])  # in this list, messages NEVER get updated
            return update_messages[1:], None
        elif len(self.messages_by_timestamp) == 1 and \
                len(self.messages_by_timestamp[update_messages[0].encounter_time]) == nb_matches:
            # we can apply simultaneous updates to all messages in this cluster and avoid splitting; do that
            old_encounters = self.messages_by_timestamp[update_messages[0].encounter_time]
            new_encounters = []
            for encounter_idx, old_encounter in enumerate(old_encounters):
                new_encounters.append(mu.create_updated_encounter_with_message(
                    encounter_message=old_encounter, update_message=update_messages[encounter_idx],
                ))
                self._real_encounter_uids.append(update_messages[encounter_idx]._sender_uid)
                self._real_encounter_times.append(update_messages[encounter_idx]._real_encounter_time)
                self._unclustered_messages.append(update_messages[encounter_idx])  # in this list, messages NEVER get updated
            self.messages_by_timestamp[update_messages[0].encounter_time] = new_encounters
            self.risk_level = new_encounters[0].risk_level
            return update_messages[nb_matches:], None
        else:
            # we lack a bunch of update messages, so we still need to split
            messages_to_transfer = self.messages_by_timestamp[update_messages[0].encounter_time][:nb_matches]
            messages_to_keep = self.messages_by_timestamp[update_messages[0].encounter_time][nb_matches:]
            updated_messages_to_transfer = [
                mu.create_updated_encounter_with_message(
                    encounter_message=messages_to_transfer[idx], update_message=update_messages[idx],
                ) for idx in range(len(messages_to_transfer))
            ]
            if not messages_to_keep:
                del self.messages_by_timestamp[update_messages[0].encounter_time]
            else:
                self.messages_by_timestamp[update_messages[0].encounter_time] = messages_to_keep
            # todo: create cluster from message batch
            new_cluster = self.create_cluster_from_message(updated_messages_to_transfer[0])
            if len(updated_messages_to_transfer) > 1:
                new_cluster._force_fit_encounter_message_batch(updated_messages_to_transfer[1:])
            return update_messages[nb_matches:], new_cluster
            # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars

    def fit_cluster(
            self,
            cluster: "NaiveCluster",
    ) -> None:
        """Updates this cluster to incorporate all the encounters in the provided cluster.

        This function will throw if anything funky is detected.

        WARNING: the cluster provided to this function must be discarded after this call!
        If this is not done, we will have duplicated messages somewhere in the manager...
        """
        # @@@@ TODO: batch-fit clusters? (will avoid multi loop+extend below)
        assert check_uids_match(self.messages_by_timestamp, cluster.messages_by_timestamp)
        assert self.risk_level == cluster.risk_level
        self.first_update_time = min(self.first_update_time, cluster.first_update_time)
        self.latest_update_time = max(self.latest_update_time, cluster.latest_update_time)
        # note: encounters should NEVER be duplicated! if these get copied here, we expect
        #       that the provided 'cluster' object will get deleted!
        for timestamp, encounters in cluster.messages_by_timestamp.items():
            if timestamp not in self.messages_by_timestamp:
                self.messages_by_timestamp[timestamp] = []
            self.messages_by_timestamp[timestamp].extend(encounters)
        # we can make sure whoever tries to use the other cluster again will have a bad surprise...
        cluster.messages_by_timestamp = None
        self._real_encounter_uids.extend(cluster._real_encounter_uids)
        self._real_encounter_times.extend(cluster._real_encounter_times)
        self._unclustered_messages.extend(cluster._unclustered_messages)

    def get_encounter_count(self):
        """Returns the total number of encounters aggregated into this cluster."""
        return sum([len(m) for m in self.messages_by_timestamp.values()])

    def get_cluster_embedding(self, include_cluster_id: bool) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        # note: this returns an array of four 'features', i.e. the cluster ID, the cluster's
        #       average encounter risk level, the number of messages in the cluster, and
        #       the first encounter timestamp of the cluster. This array's type will be returned
        #       as np.int64 to insure that no data is lost w.r.t. message counts or timestamps.
        tot_messages = self.get_encounter_count()
        if include_cluster_id:
            return np.asarray([self.cluster_id, self.risk_level,
                               tot_messages, self.first_update_time], dtype=np.int64)
        else:
            return np.asarray([self.risk_level,
                               tot_messages, self.first_update_time], dtype=np.int64)

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event)
                    for messages in self.messages_by_timestamp.values()
                    for m in messages])


class NaiveClusterManager(ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a naive clustering strategy where encounters can be combined across
    timestamps as long as their UIDs partly match and as long as their risk levels are the same.
    Update messages can also split clusters into two parts, where only one part will receive an
    update. Merging of identical clusters will happen periodically to keep the overall count low.

    THE UPDATE ALGORITHM IS NON-DETERMINISTIC. Make sure to seed your experiments if you want
    to see reproducible behavior.
    """

    clusters: typing.List[NaiveCluster]
    ticks_per_uid_roll: TimeOffsetType

    def __init__(
            self,
            ticks_per_uid_roll: TimeOffsetType = 24 * 60 * 60,  # one tick per second, one roll per day
            max_history_ticks_offset: TimeOffsetType = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
            max_cluster_id: int = 1000,  # let's hope no user ever reaches 1000 simultaneous clusters
            rng=np.random,
    ):
        super().__init__(
            max_history_ticks_offset=max_history_ticks_offset,
            add_orphan_updates_as_clusters=add_orphan_updates_as_clusters,
            generate_embeddings_by_timestamp=generate_embeddings_by_timestamp,
        )
        self.ticks_per_uid_roll = ticks_per_uid_roll
        self.rng = rng
        self.max_cluster_id = max_cluster_id
        self.next_cluster_id = 0

    def _merge_clusters(self):
        """Merges clusters that have the exact same signature (because of updates)."""
        cluster_matches, reserved_idxs_for_merge = [], []
        for base_cluster_idx, cluster in enumerate(self.clusters):
            matched_cluster_idxs = []
            for target_cluster_idx in reversed(range(base_cluster_idx + 1, len(self.clusters))):
                if target_cluster_idx in reserved_idxs_for_merge:
                    continue
                if cluster.risk_level == self.clusters[target_cluster_idx].risk_level and \
                     check_uids_match(cluster.messages_by_timestamp,
                                      self.clusters[target_cluster_idx].messages_by_timestamp):
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
            cluster_update_offset = int(current_timestamp) - int(cluster.latest_update_time)
            if cluster_update_offset <= self.max_history_ticks_offset:
                for batch_timestamp in list(cluster.messages_by_timestamp.keys()):
                    message_update_offset = int(current_timestamp) - int(batch_timestamp)
                    if message_update_offset > self.max_history_ticks_offset:
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
        # naive clustering = add encounter to any cluster that will accept it, or create a new one
        # ... to keep the clustering stochastic, we will shuffle the clusters for every message
        clusters = [c for c in self.clusters]
        self.rng.shuffle(clusters)  # ...should be a pretty quick call? right..?
        best_matched_cluster = None
        for cluster in clusters:
            score = cluster._get_encounter_match_score(message, self.ticks_per_uid_roll)
            if score > 0 and (not best_matched_cluster or best_matched_cluster[1] < score):
                best_matched_cluster = (cluster, score)
        if best_matched_cluster:
            best_matched_cluster[0]._force_fit_encounter_message(message)
        else:
            self._add_new_cluster_from_message(message)

    def _add_encounter_message_batch(self, messages: typing.List[mu.EncounterMessage], cleanup: bool = True):
        """Fits a batch of encounter messages to existing clusters, and forward the remaining to non-batch impl."""
        # for this to work, we assume that all encounters have the same uid/risks/timestamps!
        if not messages or self._check_if_message_outdated(messages[0], cleanup=False):
            return
        clusters = [c for c in self.clusters]
        self.rng.shuffle(clusters)  # ...should be a pretty quick call? right..?
        best_matched_cluster = None
        for cluster in clusters:
            score = cluster._get_encounter_match_score(messages[0], self.ticks_per_uid_roll)
            if score > 0 and (not best_matched_cluster or best_matched_cluster[1] < score):
                best_matched_cluster = (cluster, score)
        if best_matched_cluster:
            best_matched_cluster[0]._force_fit_encounter_message_batch(messages)
        else:
            self._add_new_cluster_from_message_batch(messages)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_update_message(self, message: mu.UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # update-message-to-encounter-message-matching should not be uncertain; we will
        # go through all clusters and fit the update message to the first instance that will take it
        found_adopter = False
        for cluster in self.clusters:
            if cluster.risk_level != message.old_risk_level:
                # naive clusters should always reflect the risk level of all their encounters; if
                # we can't match the risk level here, there's no way the update can apply on it
                continue
            fit_result = cluster.fit_update_message(message)
            if fit_result is None or isinstance(fit_result, NaiveCluster):
                if fit_result is not None and isinstance(fit_result, NaiveCluster):
                    fit_result.cluster_id = self.next_cluster_id
                    self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                    self.clusters.append(fit_result)
                found_adopter = True
                break
        if not found_adopter and self.add_orphan_updates_as_clusters:
            self._add_new_cluster_from_message(message)
        elif not found_adopter:
            raise AssertionError(f"could not find adopter for: {message}")

    def _add_update_message_batch(self, messages: typing.List[mu.UpdateMessage], cleanup: bool = True):
        """Fits a batch of update messages to existing clusters, and forwards the remaining to non-batch impl."""
        # for this to work, we assume that all encounters have the same uid/risks/timestamps!
        if not messages or self._check_if_message_outdated(messages[0], cleanup=False):
            return
        for cluster in self.clusters:
            if cluster.risk_level != messages[0].old_risk_level:
                # naive clusters should always reflect the risk level of all their encounters; if
                # we can't match the risk level here, there's no way the update can apply on it
                continue
            messages, new_cluster = cluster.fit_update_message_batch(messages)
            if new_cluster is not None:
                new_cluster.cluster_id = self.next_cluster_id
                self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                self.clusters.append(new_cluster)
            if not messages:
                break  # all messages got adopted
        if messages and self.add_orphan_updates_as_clusters:
            self._add_new_cluster_from_message_batch(messages)
        elif messages:
            raise AssertionError(f"could not find adopters for {len(messages)}x of: {messages[0]}")
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_new_cluster_from_message(self, message: mu.GenericMessageType):
        """Creates and adds a new cluster in the internal structs while cycling the cluster ids."""
        new_cluster = NaiveCluster.create_cluster_from_message(message, self.next_cluster_id)
        self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
        self.clusters.append(new_cluster)

    def _add_new_cluster_from_message_batch(self, messages: typing.List[mu.GenericMessageType]):
        """Creates and adds a new cluster in the internal structs while cycling the cluster ids."""
        if not messages:
            return
        if isinstance(messages[0], mu.EncounterMessage):
            new_cluster = NaiveCluster.create_cluster_from_message(messages[0], self.next_cluster_id)
            new_cluster._force_fit_encounter_message_batch(messages[1:])
        else:
            assert self.add_orphan_updates_as_clusters
            new_cluster = NaiveCluster.create_cluster_from_message(messages[0], self.next_cluster_id)
            new_encounters = [mu.create_encounter_from_update_message(m) for m in messages[1:]]
            new_cluster._force_fit_encounter_message_batch(new_encounters)
        self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
        self.clusters.append(new_cluster)

    def get_embeddings_array(self) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        if self.generate_embeddings_by_timestamp:
            cluster_embeds = collections.defaultdict(list)
            for cluster in self.clusters:
                embed = cluster.get_cluster_embedding(include_cluster_id=True)
                for timestamp in cluster.messages_by_timestamp:
                    cluster_embeds[timestamp].append(embed)
            flat_output = []
            for timestamp in sorted(cluster_embeds.keys()):
                flat_output.extend(cluster_embeds[timestamp])
            return np.asarray(flat_output)
        else:
            return np.asarray([c.get_cluster_embedding(include_cluster_id=False)
                               for c in self.clusters], dtype=np.int64)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        if self.generate_embeddings_by_timestamp:
            cluster_flags = collections.defaultdict(list)
            for cluster in self.clusters:
                flags = cluster._get_cluster_exposition_flag()
                for timestamp in cluster.messages_by_timestamp:
                    cluster_flags[timestamp].append(flags)
            flat_output = []
            for timestamp in sorted(cluster_flags.keys()):
                flat_output.extend(cluster_flags[timestamp])
            return np.asarray(flat_output)
        else:
            return np.asarray([c._get_cluster_exposition_flag() for c in self.clusters], dtype=np.uint8)
