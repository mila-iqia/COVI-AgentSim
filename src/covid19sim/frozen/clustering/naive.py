import numpy as np
import typing

import covid19sim.frozen.message_utils as mu
import covid19sim.frozen.clustering.base as base
import covid19sim.frozen.clustering.simple as clu


def check_uids_match(
        uids1: typing.Dict[np.int64, np.uint8],
        uids2: typing.Dict[np.int64, np.uint8],
) -> bool:
    """Returns whether the overlaps in the provided timestamp-to-uid dicts are compatible.

    Note: if there are no overlapping UIDs in the two sets, the function will return `False`.
    """
    overlapping_timestamps = list(set(uids1.keys()) & set(uids2.keys()))
    return overlapping_timestamps and all([uids1[t] == uids2[t] for t in overlapping_timestamps])


class NaiveCluster(clu.SimpleCluster):
    """A naive message cluster.

    The default implementation of the 'fit' functions for this base class will
    attempt to merge new messages across uid rolls and create new clusters if the risk
    level of the cluster is not uniform for all messages it aggregates.
    """

    # note: the 'naive cluster' no longer uses a global uid, but this dict instead
    # (...the use of that uid for anything should therefore be discouraged, as it will
    #  only correspond to the first encounter uid it receives, and it never gets updated)
    uids: typing.Dict[np.int64, np.uint8]
    """Timestamp-to-Unique Identifier (UID) mapping of all encounters in this cluster."""

    def __init__(
            self,
            uid: np.uint8,
            first_update_time: np.int64,
            **kwargs,
    ):
        """Creates a naive cluster, forwarding most args to the base class."""
        super().__init__(
            uid=uid,
            first_update_time=first_update_time,
            **kwargs,
        )
        self.uids = {first_update_time: uid}

    @staticmethod
    def create_cluster_from_message(message: mu.GenericMessageType) -> "NaiveCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return NaiveCluster(
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

    def _get_encounter_match_score(
            self,
            message: mu.EncounterMessage,
            ticks_per_uid_roll: np.int64 = 24 * 60 * 60,  # one tick per second, one roll per day
    ) -> typing.Tuple[int, typing.Optional[int]]:
        """Returns the match score between the provided encounter message and this cluster.

        A negative return value means the message & cluster cannot correspond to the same person. A
        zero value means that no match can be established with certainty (e.g. because the message
        or cluster is too old). A positive value indicates a (possibly partial) match. A high
        positive value means the match is more likely. If the returned value is equal to
        `message_uid_bit_count`, the match is perfect.

        Returns:
            A tuple of the best match score (an integer in `[-1,message_uid_bit_count]`) and of
            the matched encounter index (for internal use).
        """
        if message.risk_level != self.risk_level:
            # if the cluster's risk level differs from the given encounter's risk level, there is
            # no way to match the two, as the cluster should be updated first to that risk level,
            # or we are looking at different users entirely
            return -1, None
        if message.encounter_time in self.uids and self.uids[message.encounter_time] != message.uid:
            # if we already have a stored uid at the new encounter's timestamp with a different
            # uid, there's no way this message can be merged into this cluster
            return -1, None
        best_match = (-1, None)
        assert len(self.messages), "... a cluster cannot exist without past encounters?"
        for encounter_idx, old_encounter in enumerate(self.messages):
            match_score = mu.find_encounter_match_score(
                # TODO: what happens if the received message is actually late, and we have an
                #       'old message' that is more recent? (match scorer will throw)
                old_encounter,
                message,
                ticks_per_uid_roll,
            )
            if match_score > best_match[0]:
                best_match = (match_score, encounter_idx)
        return best_match

    def _force_fit_encounter_message(
            self,
            message: mu.EncounterMessage,
            internal_encounter_idx: int,
    ) -> typing.Optional[mu.EncounterMessage]:
        """Updates the current cluster given a new encounter message.

        Same as `fit_encounter_message`, but allows an internal message index to be passed in to
        avoid an extra `_get_encounter_match_score` call.
        """
        # do something with internal_encounter_idx? (we don't actually need it in this version)
        assert internal_encounter_idx < len(self.messages)
        # update the cluster time with the new message's encounter time (if more recent)
        self.latest_update_time = max(message.encounter_time, self.latest_update_time)
        self.messages.append(message)  # in this list, encounters may get updated (and form new clusters)
        self.uids[message.encounter_time] = message.uid
        self._real_encounter_uids.append(message._sender_uid)
        self._real_encounter_times.append(message._real_encounter_time)
        self._unclustered_messages.append(message)  # in this list, messages NEVER get updated
        return None  # we merged the message in, so nothing new to return to the manager

    def fit_encounter_message(
            self,
            message: mu.EncounterMessage,
            ticks_per_uid_roll: np.int64 = 24 * 60 * 60,  # one tick per second, one roll per day
            minimum_match_score: int = 1,  # means we should at least find a 1-bit uid match
    ) -> typing.Optional[mu.EncounterMessage]:
        """Updates the current cluster given a new encounter message.

        If the already-clustered encounter messages cannot be at least partially matched with the
        new encounter, the given message will be returned as-is. Otherwise, it will be added to
        the cluster, and the function will return `None`.
        """
        assert message.risk_level == self.risk_level, "cluster and new encounter message risks mismatch"
        best_match_score, best_match_idx = self._get_encounter_match_score(
            message=message,
            ticks_per_uid_roll=ticks_per_uid_roll,
        )
        if best_match_score < minimum_match_score:
            # ask the manager to add the message as a new cluster instead of merging it in
            return message
        return self._force_fit_encounter_message(message=message, internal_encounter_idx=best_match_idx)

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
        # TODO: could do a batch-fit-update to avoid splitting and merging a lot of clusters in manager
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_message.old_risk_level == self.risk_level, "cluster & update message old risk mismatch"
        # quick-exit: if this cluster does not contain the timestamp for the encounter, or if the
        # cluster contains a different uid for that timestamp, there is no way this message is compatible
        if update_message.encounter_time not in self.uids or \
                self.uids[update_message.encounter_time] != update_message.uid:
            # could not find any match for the update message; send it back to the manager
            return update_message
        found_match = None
        for old_encounter_idx, old_encounter in enumerate(self.messages):
            assert update_message.old_risk_level == old_encounter.risk_level
            if old_encounter.uid == update_message.uid and \
                    old_encounter.encounter_time == update_message.encounter_time:
                found_match = (old_encounter_idx, old_encounter)
                break
        if found_match is not None:
            if len(self.messages) == 1:
                # we can self-update without splitting; do that
                assert found_match[0] == 0
                self.messages[0] = mu.create_updated_encounter_with_message(
                    encounter_message=self.messages[0], update_message=update_message,
                )
                self.risk_level = self.messages[0].risk_level
                self._real_encounter_uids.append(update_message._sender_uid)
                self._real_encounter_times.append(update_message._real_encounter_time)
                self._unclustered_messages.append(update_message)  # in this list, messages NEVER get updated
                return None
            else:
                # we have multiple messages in this cluster, and the update can only apply to one;
                # ... we need to split the cluster into two, where only the new one will be updated
                return self.create_cluster_from_message(mu.create_updated_encounter_with_message(
                    encounter_message=self.messages.pop(found_match[0]), update_message=update_message,
                ))
                # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars
        else:
            # could not find any match for the update message; send it back to the manager
            return update_message

    def fit_cluster(
            self,
            cluster: "NaiveCluster",
    ) -> None:
        """Updates this cluster to incorporate all the encounters in the provided cluster.

        This function will throw if anything funky is detected.

        WARNING: the cluster provided to this function must be discarded after this call!
        If this is not done, we will have duplicated messages somewhere in the manager...
        """
        assert check_uids_match(self.uids, cluster.uids)
        assert self.risk_level == cluster.risk_level
        self.first_update_time = min(self.first_update_time, cluster.first_update_time)
        self.latest_update_time = max(self.latest_update_time, cluster.latest_update_time)
        # note: encounters should NEVER be duplicated! if these get copied here, we expect
        #       that the provided 'cluster' object will get deleted!
        self.messages.extend(cluster.messages)
        # we can make sure whoever tries to use the cluster again will have a bad surprise...
        cluster.messages = None
        self._real_encounter_uids.extend(cluster._real_encounter_uids)
        self._real_encounter_times.extend(cluster._real_encounter_times)
        self._unclustered_messages.extend(cluster._unclustered_messages)

    def get_cluster_embedding(self) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        # note: this returns an array of four 'features', i.e. the cluster UID, the cluster's
        #       constant encounter risk level, the number of messages in the cluster, and
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


class NaiveClusterManager(base.ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a naive clustering strategy where encounters can be combined across
    timestamps as long as their UIDs partly match and as long as their risk levels are the same.
    Update messages can also split clusters into two parts, where only one part will receive an
    update. Merging of identical clusters will happen periodically to keep the overall count low.

    THE UPDATE ALGORITHM IS NON-DETERMINISTIC. Make sure to seed your experiments if you want
    to see reproducible behavior.
    """

    clusters: typing.List[NaiveCluster]
    ticks_per_uid_roll: np.int64
    add_orphan_updates_as_clusters: bool

    def __init__(
            self,
            max_history_ticks_offset: np.int64 = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            ticks_per_uid_roll: np.int64 = 24 * 60 * 60,  # one tick per second, one roll per day
            add_orphan_updates_as_clusters: bool = False,
            rng=np.random,
    ):
        super().__init__(max_history_ticks_offset=max_history_ticks_offset)
        self.ticks_per_uid_roll = ticks_per_uid_roll
        self.add_orphan_updates_as_clusters = add_orphan_updates_as_clusters
        self.rng = rng

    def _merge_clusters(self):
        """Merges clusters that have the exact same signature (because of updates)."""
        # TODO: could do a batch-fit-update to avoid splitting and merging a lot of clusters every day
        cluster_matches, reserved_idxs_for_merge = [], []
        for base_cluster_idx, cluster in enumerate(self.clusters):
            matched_cluster_idxs = []
            for target_cluster_idx in reversed(range(base_cluster_idx + 1, len(self.clusters))):
                if target_cluster_idx in reserved_idxs_for_merge:
                    continue
                if cluster.risk_level == self.clusters[target_cluster_idx].risk_level and \
                     check_uids_match(cluster.uids, self.clusters[target_cluster_idx].uids):
                    matched_cluster_idxs.append(target_cluster_idx)
            cluster_matches.append(matched_cluster_idxs)
            reserved_idxs_for_merge.extend(matched_cluster_idxs)
        to_keep = []
        for base_cluster_idx, (cluster, target_idxs) in enumerate(zip(self.clusters, cluster_matches)):
            if base_cluster_idx in reserved_idxs_for_merge:
                continue
            for target_idx in target_idxs:
                cluster.fit_cluster(self.clusters[target_idx])
            to_keep.append(cluster)
        self.clusters = to_keep

    def add_messages(self, messages: typing.Iterable[mu.GenericMessageType], cleanup: bool = True):
        """Dispatches the provided messages to the correct internal 'add' function based on type."""
        for message in messages:
            if isinstance(message, mu.EncounterMessage):
                self._add_encounter_message(message, cleanup=False)
            elif isinstance(message, mu.UpdateMessage):
                self._add_update_message(message, cleanup=False)
            else:
                ValueError("unexpected message type")
        self._merge_clusters()  # there's a bit of looping in here, think about batching?
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
            score, encounter_idx = cluster._get_encounter_match_score(message, self.ticks_per_uid_roll)
            # @@@ merge = what min score?
            if score > 0 and (not best_matched_cluster or best_matched_cluster[1] < score):
                best_matched_cluster = (cluster, score, encounter_idx)
        if best_matched_cluster:
            best_matched_cluster[0]._force_fit_encounter_message(message, best_matched_cluster[2])
        else:
            self.clusters.append(NaiveCluster.create_cluster_from_message(message))

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
                    self.clusters.append(fit_result)
                found_adopter = True
                break
        if not found_adopter and self.add_orphan_updates_as_clusters:
            self.clusters.append(NaiveCluster.create_cluster_from_message(message))
        elif not found_adopter:
            raise AssertionError(f"could not find any proper cluster match for: {message}")
