import typing

from covid19sim.frozen.message_utils import EncounterMessage, UpdateMessage
from covid19sim.frozen.clustering.base import ClusterIDType, TimestampType, TimeOffsetType, \
    ClusterManagerBase
from covid19sim.frozen.clustering.simple import SimpleCluster


class PerfectClusterManager(ClusterManagerBase):
    """Manages message cluster creation and updates.

    This class implements a perfect clustering strategy where encounters will be combined
    using the UNOBSERVED variables inside messages. This means that this algorithm should only
    be used for experimental evaluation purposes, AND IT CANNOT ACTUALLY WORK IN PRACTICE.
    """

    clusters: typing.List[SimpleCluster]

    def __init__(
            self,
            max_history_ticks_offset: TimeOffsetType = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
    ):
        assert not add_orphan_updates_as_clusters, "missing impl; this is... imperfect?"
        super().__init__(
            max_history_ticks_offset=max_history_ticks_offset,
            add_orphan_updates_as_clusters=add_orphan_updates_as_clusters,
            generate_embeddings_by_timestamp=generate_embeddings_by_timestamp,
        )
        # note: no RNG here, this impl is deterministic

    def _add_encounter_message(self, message: EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # perfect clustering = we are looking for an exact timestamp/realuid/risk level match;
        # if one is not found, we create a new cluster
        assert message._sender_uid is not None
        matched_clusters = []
        for cluster in self.clusters:
            # note: all 'real' uids in the cluster should be the same, we just pick the first
            if cluster._real_encounter_uids[0] == message._sender_uid:
                matched_clusters.append(cluster)
        if matched_clusters:
            # this is perfect clustering; there should always be a single cluster per user?
            assert len(matched_clusters) == 1
            matched_clusters[0].fit_encounter_message(message)
        else:
            new_cluster = SimpleCluster.create_cluster_from_message(message)
            self.clusters.append(new_cluster)

    def _add_update_message(self, message: UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # perfect clustering = we will update the encounters in the user's cluster one
        # update message at a time, and hope that no update messages will be missing
        assert message._sender_uid is not None
        matched_clusters = []
        for cluster in self.clusters:
            # note: all 'real' uids in the cluster should be the same, we just pick the first
            if cluster._real_encounter_uids[0] == message._sender_uid:
                assert any([m.risk_level == message.old_risk_level for m in cluster.messages])
                matched_clusters.append(cluster)
        if matched_clusters:
            # this is perfect clustering; there should always be a single cluster per user?
            assert len(matched_clusters) == 1
            res = matched_clusters[0].fit_update_message(message)
            assert res is None
        else:
            raise AssertionError(f"could not find any proper cluster match for: {message}")
