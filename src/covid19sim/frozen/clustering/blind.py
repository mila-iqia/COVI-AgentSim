import dataclasses
import numpy as np
import typing

from covid19sim.frozen.message_utils import EncounterMessage, GenericMessageType, UpdateMessage, \
    TimestampType, create_encounter_from_update_message, create_updated_encounter_with_message
from covid19sim.frozen.clustering.base import ClusterIDType, RealUserIDType, TimeOffsetType, \
    ClusterBase, ClusterManagerBase, MessagesArrayType
from covid19sim.frozen.clustering.simple import SimpleCluster, SimplisticClusterManager

# TODO: determine whether these class can really derive from their 'simple' counterparts, and
#       make sure there is no bad stuff happening when calling functions from outside


class BlindCluster(ClusterBase):
    """A blind encounter message cluster.

    The default implementation of the 'fit' functions for this base class will
    simply attempt to merge new messages if they share a risk level and split the cluster
    if partial updates are received. This implementation does not rely on the sender UID.

    This cluster can also only contain encounters from a single timestamp.
    """

    messages: typing.List[EncounterMessage] = dataclasses.field(default_factory=list)
    """List of encounter messages aggregated into this cluster (in added order)."""
    # note: messages above might have been updated from their original state, but they
    #       should all share the same risk level (partial updates will split the cluster)

    def __init__(
            self,
            messages: typing.List[EncounterMessage],
            **kwargs,
    ):
        """Creates a simple cluster, forwarding most args to the base class."""
        super().__init__(
            **kwargs,
        )
        self.messages = messages

    @staticmethod
    def create_cluster_from_message(
            message: GenericMessageType,
            cluster_id: ClusterIDType,
    ) -> "BlindCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return BlindCluster(
            # app-visible stuff below
            cluster_id=cluster_id,
            risk_level=message.risk_level
                if isinstance(message, EncounterMessage) else message.new_risk_level,
            first_update_time=message.encounter_time,
            latest_update_time=message.encounter_time,
            messages=[message] if isinstance(message, EncounterMessage)
                else [create_encounter_from_update_message(message)],
            # debug-only stuff below
            _real_encounter_uids={message._sender_uid},
            _real_encounter_times={message._real_encounter_time},
        )

    def fit_encounter_message(self, message: EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        # @@@@ TODO: batch-fit encounters? (will avoid multi loop+extend below)
        assert message.risk_level == self.risk_level
        assert message.encounter_time == self.first_update_time
        assert message.encounter_time == self.latest_update_time
        self.messages.append(message)  # in this list, encounters may get updated
        self._real_encounter_uids.add(message._sender_uid)
        self._real_encounter_times.add(message._real_encounter_time)

    def fit_update_message(
            self,
            update_message: UpdateMessage
    ) -> typing.Optional[typing.Union[UpdateMessage, "BlindCluster"]]:
        """Updates an encounter in the current cluster given a new update message.

        If this cluster gets split as a result of the update, the function will return the newly
        created cluster. Otherwise, if the update message cannot be applied to any encounter in this
        cluster, it will be returned as-is. Finally, if the update message was applied to the cluster
        without splitting it, the function will return `None`.
        """
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_message.old_risk_level == self.risk_level
        assert update_message.encounter_time == self.first_update_time
        assert update_message.encounter_time == self.latest_update_time
        if len(self.messages) == 1:
            # we can self-update without splitting; do that
            self.messages[0] = create_updated_encounter_with_message(
                encounter_message=self.messages[0], update_message=update_message,
                blind_update=True,
            )
            self.risk_level = update_message.new_risk_level
            self._real_encounter_uids.add(update_message._sender_uid)
            self._real_encounter_times.add(update_message._real_encounter_time)
            return None
        else:
            # we have multiple messages in this cluster, and the update can only apply to one;
            # ... we need to split the cluster into two, where only the new one will be updated
            return self.create_cluster_from_message(create_updated_encounter_with_message(
                encounter_message=self.messages.pop(0), update_message=update_message,
                blind_update=True,
            ), cluster_id=None)  # cluster id will be assigned properly in manager
            # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars

    def fit_update_message_batch(
            self,
            update_messages: typing.List[UpdateMessage],
    ) -> typing.Tuple[typing.List[UpdateMessage], typing.Optional["BlindCluster"]]:
        """Updates encounters in the current cluster given a list of new update messages.

        If this cluster gets split as a result of the update, the function will return the newly
        created cluster. Otherwise, if the update messages cannot be applied to any encounter in this
        cluster, they will be returned as-is. Finally, if any update message was applied to the cluster
        without splitting it, the function will return the remaining updates.
        """
        if not update_messages:
            return [], None
        # TODO: what will happen when update messages are no longer systematically sent? (assert will break)
        assert update_messages[0].old_risk_level == self.risk_level
        assert update_messages[0].encounter_time == self.first_update_time
        assert update_messages[0].encounter_time == self.latest_update_time
        nb_matches = min(len(update_messages), len(self.messages))
        if len(self.messages) == 1 and nb_matches == 1:
            # we can trivially self-update without splitting; do that
            self.messages[0] = create_updated_encounter_with_message(
                encounter_message=self.messages[0], update_message=update_messages[0],
                blind_update=True,
            )
            self.risk_level = self.messages[0].risk_level
            self._real_encounter_uids.add(update_messages[0]._sender_uid)
            self._real_encounter_times.add(update_messages[0]._real_encounter_time)
            return update_messages[1:], None
        elif len(self.messages) == nb_matches:
            # we can apply simultaneous updates to all messages in this cluster and avoid splitting; do that
            new_encounters = []
            for encounter_idx, old_encounter in enumerate(self.messages):
                new_encounters.append(create_updated_encounter_with_message(
                    encounter_message=old_encounter, update_message=update_messages[encounter_idx],
                    blind_update=True,
                ))
            self.messages = new_encounters
            self._real_encounter_uids.update([m._sender_uid for m in update_messages[:nb_matches]])
            self._real_encounter_times.update([m._real_encounter_time for m in update_messages[:nb_matches]])
            self.risk_level = new_encounters[0].risk_level
            return update_messages[nb_matches:], None
        else:
            # we lack a bunch of update messages, so we still need to split
            messages_to_transfer = self.messages[:nb_matches]
            messages_to_keep = self.messages[nb_matches:]
            updated_messages_to_transfer = [
                create_updated_encounter_with_message(
                    encounter_message=messages_to_transfer[idx], update_message=update_messages[idx],
                    blind_update=True,
                ) for idx in range(len(messages_to_transfer))
            ]
            assert messages_to_keep
            self.messages = messages_to_keep
            new_cluster = BlindCluster(
                cluster_id=None,  # cluster id will be assigned properly in manager
                risk_level=updated_messages_to_transfer[0].risk_level,
                first_update_time=updated_messages_to_transfer[0].encounter_time,
                latest_update_time=updated_messages_to_transfer[0].encounter_time,
                messages=updated_messages_to_transfer,
                _real_encounter_uids={m._sender_uid for m in updated_messages_to_transfer},
                _real_encounter_times={m._real_encounter_time for m in updated_messages_to_transfer},
            )
            return update_messages[nb_matches:], new_cluster
            # note: out of laziness for the debugging stuff, we do not remove anything from unobserved vars

    def fit_cluster(
            self,
            cluster: "BlindCluster",
    ) -> None:
        """Updates this cluster to incorporate all the encounters in the provided cluster.

        This function will throw if anything funky is detected.

        WARNING: the cluster provided to this function must be discarded after this call!
        If this is not done, we will have duplicated messages somewhere in the manager...
        """
        # @@@@ TODO: batch-fit clusters? (will avoid multi loop+extend below)
        assert self.risk_level == cluster.risk_level
        assert self.first_update_time == cluster.first_update_time
        assert self.latest_update_time == cluster.latest_update_time
        self.messages.extend(cluster.messages)
        # we can make sure whoever tries to use the other cluster again will have a bad surprise...
        cluster.messages = None
        self._real_encounter_uids.update(cluster._real_encounter_uids)
        self._real_encounter_times.update(cluster._real_encounter_times)

    def get_cluster_embedding(
            self,
            current_timestamp: TimestampType,
            include_cluster_id: bool,
            old_compat_mode: bool = False,
    ) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        # code is 100% identical in SimpleCluster, use that instead
        return SimpleCluster.get_cluster_embedding(
            self,
            current_timestamp=current_timestamp,
            include_cluster_id=include_cluster_id,
            old_compat_mode=old_compat_mode,
        )

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # code is 100% identical in SimpleCluster, use that instead
        return SimpleCluster._get_cluster_exposition_flag(self)

    def get_timestamps(self) -> typing.List[TimestampType]:
        """Returns the list of timestamps for which this cluster possesses at least one encounter."""
        return [self.first_update_time]  # this impl's clusters always only cover a single timestamp

    def get_encounter_count(self) -> int:
        """Returns the number of encounters aggregated inside this cluster."""
        return len(self.messages)


class BlindClusterManager(ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a blind clustering strategy where encounters are only combined
    on a timestamp and risk-level basis. This means clusters cannot contain messages with
    different timestamps or risk levels.
    """

    clusters: typing.List[BlindCluster]

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
                if cluster.risk_level == self.clusters[target_cluster_idx].risk_level and \
                        cluster.first_update_time == self.clusters[target_cluster_idx].first_update_time:
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
        """Gets rid of clusters that are too old given the current timestamp."""
        to_keep = []
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_update_offset = current_timestamp - cluster.first_update_time
            if cluster_update_offset < self.max_history_offset:
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

    def _add_encounter_message(self, message: EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # blind clustering = we are looking for an exact timestamp/risk level match
        matched_cluster = None
        for cluster in self.clusters:
            if cluster.risk_level == message.risk_level and \
                    cluster.first_update_time == message.encounter_time:
                matched_cluster = cluster
                break
        if matched_cluster is not None:
            matched_cluster.fit_encounter_message(message)
        else:
            new_cluster = BlindCluster.create_cluster_from_message(message, self.next_cluster_id)
            self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
            self.clusters.append(new_cluster)

    def _add_update_message(self, message: UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        matched_cluster_idx = None
        for cluster_idx, cluster in enumerate(self.clusters):
            if cluster.risk_level == message.old_risk_level and \
                    cluster.first_update_time == message.encounter_time:
                matched_cluster_idx = cluster_idx
                break
        if matched_cluster_idx is not None:
            fit_result = self.clusters[matched_cluster_idx].fit_update_message(message)
            if fit_result is not None:
                assert isinstance(fit_result, BlindCluster)
                fit_result.cluster_id = self.next_cluster_id
                self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                self.clusters.insert(matched_cluster_idx + 1, fit_result)
        else:
            if self.add_orphan_updates_as_clusters:
                new_cluster = BlindCluster.create_cluster_from_message(message, self.next_cluster_id)
                self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
                self.clusters.append(new_cluster)
            else:
                raise AssertionError(f"could not find any proper cluster match for: {message}")

    def get_embeddings_array(
            self,
            cleanup: bool = False,
            current_timestamp: typing.Optional[TimestampType] = None,  # will use internal latest if None
    ) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        # code is 100% identical in SimplisticClusterManager, use that instead
        return SimplisticClusterManager.get_embeddings_array(self, cleanup, current_timestamp)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        # code is 100% identical in SimplisticClusterManager, use that instead
        return SimplisticClusterManager._get_expositions_array(self)

    def _get_homogeneity_scores(self) -> typing.Dict[RealUserIDType, float]:
        """Returns the homogeneity score for all real users in the clusters."""
        # code is 100% identical in SimplisticClusterManager, use that instead
        return SimplisticClusterManager._get_homogeneity_scores(self)

    def get_encounters_cluster_mapping(self) -> typing.List[typing.Tuple[EncounterMessage, ClusterIDType]]:
        """Returns a flattened list of encounters mapped to their cluster ids."""
        return [(encounter, c.cluster_id) for c in self.clusters for encounter in c.messages]

    def _get_cluster_count_error(self) -> int:
        """Returns the difference between the number of clusters and the number of unique users."""
        # code is 100% identical in SimplisticClusterManager, use that instead
        return SimplisticClusterManager._get_cluster_count_error(self)
