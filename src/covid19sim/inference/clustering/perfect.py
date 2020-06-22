from covid19sim.inference.message_utils import EncounterMessage, UpdateMessage
from covid19sim.inference.clustering.base import TimeOffsetType
from covid19sim.inference.clustering.simple import SimpleCluster, SimplisticClusterManager


class PerfectClusterManager(SimplisticClusterManager):
    """Manages message cluster creation and updates.

    This class implements a perfect clustering strategy where encounters will be combined
    using the UNOBSERVED variables inside messages. This means that this algorithm should only
    be used for experimental evaluation purposes, AND IT CANNOT ACTUALLY WORK IN PRACTICE.
    """

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

    def _add_encounter_message(self, message: EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # perfect clustering = we are looking for an exact real-user-id match;
        # if one is not found, we create a new cluster
        assert message._sender_uid is not None
        matched_cluster = None
        for cluster in self.clusters:
            if message._sender_uid in cluster._real_encounter_uids:
                matched_cluster = cluster
                break
        if matched_cluster is not None:
            matched_cluster.skip_homogeneity_checks = True  # ensure the fit won't throw
            matched_cluster.fit_encounter_message(message)
        else:
            new_cluster = SimpleCluster.create_cluster_from_message(message, self.next_cluster_id)
            self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
            self.clusters.append(new_cluster)

    def _add_update_message(self, message: UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # perfect clustering = we will update the encounters in the user's cluster one
        # update message at a time, and hope that no update messages will be missing
        assert message._sender_uid is not None
        for cluster in self.clusters:
            if message._sender_uid in cluster._real_encounter_uids:
                cluster.skip_homogeneity_checks = True  # ensure the fit won't throw
                message = cluster.fit_update_message(message)
                if message is None:
                    return
        if self.add_orphan_updates_as_clusters:
            new_cluster = SimpleCluster.create_cluster_from_message(message, self.next_cluster_id)
            self.next_cluster_id = (self.next_cluster_id + 1) % self.max_cluster_id
            self.clusters.append(new_cluster)
        else:
            raise AssertionError(f"could not find any proper cluster match for: {message}")
