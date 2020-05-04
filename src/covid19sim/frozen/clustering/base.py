import numpy as np
import typing

import covid19sim.frozen.message_utils as mu


class ClusterManagerBase:
    """Manages message cluster creation and updates.

    This base class implements common utility functions used by other clustering algos.
    """

    clusters: typing.List
    max_history_ticks_offset: int
    latest_refresh_timestamp: np.int64

    def __init__(
            self,
            max_history_ticks_offset: int = 24 * 60 * 60 * 14,  # one tick per second, 14 days
    ):
        self.clusters = []
        self.max_history_ticks_offset = max_history_ticks_offset
        self.latest_refresh_timestamp = np.int64(0)

    def cleanup_clusters(self, current_timestamp: np.int64):
        """Gets rid of clusters that are too old given the current timestamp."""
        to_keep = []
        for cluster_idx, cluster in enumerate(self.clusters):
            update_offset = int(current_timestamp) - int(cluster.latest_update_time)
            if update_offset <= self.max_history_ticks_offset:
                to_keep.append(cluster)
        self.clusters = to_keep

    def _check_if_message_outdated(self, message: mu.GenericMessageType, cleanup: bool = True) -> bool:
        """Returns whether a message is outdated or not. Will also refresh the internal check timestamp."""
        self.latest_refresh_timestamp = max(message.encounter_time, self.latest_refresh_timestamp)
        outdated = False
        if self.latest_refresh_timestamp:
            min_offset = int(self.latest_refresh_timestamp) - int(message.encounter_time)
            if min_offset > self.max_history_ticks_offset:
                # there's no way this message is useful if we get here, since it's so old
                outdated = True
            if cleanup:
                self.cleanup_clusters(self.latest_refresh_timestamp)
        return outdated

    def add_messages(self, messages: typing.Iterable[mu.GenericMessageType], cleanup: bool = True):
        """Dispatches the provided messages to the correct internal 'add' function based on type."""
        for message in messages:
            if isinstance(message, mu.EncounterMessage):
                self._add_encounter_message(message, cleanup=False)
            elif isinstance(message, mu.UpdateMessage):
                self._add_update_message(message, cleanup=False)
            else:
                ValueError("unexpected message type")
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_encounter_message(self, message: mu.EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        return NotImplementedError

    def _add_update_message(self, message: mu.UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        return NotImplementedError

    def get_embeddings_array(self) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        return np.asarray([c.get_cluster_embedding() for c in self.clusters], dtype=np.int64)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        return np.asarray([c._get_cluster_exposition_flag() for c in self.clusters], dtype=np.uint8)
