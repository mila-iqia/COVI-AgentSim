import dataclasses
import datetime
import numpy as np
import typing

from covid19sim.inference.message_utils import EncounterMessage, GenericMessageType, UpdateMessage, \
    TimestampType, TimeOffsetType, RealUserIDType, RiskLevelType, TimestampDefault

ClusterIDType = int
MessagesArrayType = typing.Union[typing.Iterable[GenericMessageType],
                                 typing.Iterable[typing.List[GenericMessageType]],
                                 typing.Iterable[typing.Dict[TimestampType, typing.List[GenericMessageType]]]]

UpdateMessageBatchType = typing.Union[typing.List[UpdateMessage],
                                      typing.Dict[TimestampType, typing.Iterable[UpdateMessage]]]


@dataclasses.dataclass
class ClusterBase:
    """An encounter message cluster."""

    cluster_id: ClusterIDType
    """Unique Identifier (UID) of the cluster."""

    risk_level: RiskLevelType
    """Quantified risk level of the cluster."""

    first_update_time: TimestampType
    """Cluster creation timestamp (i.e. timestamp of first encounter)."""

    latest_update_time: TimestampType
    """Latest cluster update timestamp (i.e. timestamp of latest encounter)."""

    ##########################################
    # private variables (for debugging only!)

    _real_encounter_uids: typing.Set[RealUserIDType] = dataclasses.field(default_factory=set)
    """Real Unique Identifiers (UIDs) of the clustered user(s)."""

    _real_encounter_times: typing.Set[TimestampType] = dataclasses.field(default_factory=set)
    """Real timestamp of the clustered encounter(s)."""

    def _is_homogenous(self) -> bool:
        """Returns whether this cluster is truly homogenous (i.e. tied to one user) or not."""
        return len(self._real_encounter_uids) <= 1

    @staticmethod
    def create_cluster_from_message(
            message: GenericMessageType,
            cluster_id: ClusterIDType,
    ):
        """Creates and returns a new cluster based on a single encounter message."""
        raise NotImplementedError

    def fit_encounter_message(self, message: EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        raise NotImplementedError

    def fit_update_message(self, update_message: UpdateMessage):
        """Updates an encounter in the current cluster given a new update message."""
        raise NotImplementedError

    def get_cluster_embedding(
            self,
            current_timestamp: TimestampType,
            include_cluster_id: bool,
            old_compat_mode: bool = False,
    ) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        raise NotImplementedError

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        raise NotImplementedError

    def get_timestamps(self) -> typing.List[TimestampType]:
        """Returns the list of timestamps for which this cluster possesses at least one encounter."""
        raise NotImplementedError

    def get_encounter_count(self) -> int:
        """Returns the number of encounters aggregated inside this cluster."""
        raise NotImplementedError


class ClusterManagerBase:
    """Manages message cluster creation and updates.

    This base class implements common utility functions used by other clustering algos.
    """

    clusters: typing.List[ClusterBase]
    latest_refresh_timestamp: TimestampType
    max_history_offset: TimeOffsetType
    add_orphan_updates_as_clusters: bool
    generate_embeddings_by_timestamp: bool
    generate_backw_compat_embeddings: bool
    max_cluster_id: int

    def __init__(
            self,
            max_history_offset: TimeOffsetType,
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
            generate_backw_compat_embeddings: bool = False,
            max_cluster_id: int = 1000,
    ):
        self._is_being_used = False  # for multithread sanity checks (could be a mutex?)
        self.clusters = []
        self.max_cluster_id = max_cluster_id
        self.next_cluster_id = 0
        self.latest_refresh_timestamp = TimestampDefault
        self.max_history_offset = max_history_offset
        self.add_orphan_updates_as_clusters = add_orphan_updates_as_clusters
        self.generate_embeddings_by_timestamp = generate_embeddings_by_timestamp
        self.generate_backw_compat_embeddings = generate_backw_compat_embeddings
        assert not self.generate_backw_compat_embeddings or self.generate_embeddings_by_timestamp

    def cleanup_clusters(self, current_timestamp: TimestampType):
        """Gets rid of clusters that are too old given the current timestamp."""
        to_keep = []
        for cluster in self.clusters:
            update_offset = current_timestamp - cluster.first_update_time
            if update_offset < self.max_history_offset:
                to_keep.append(cluster)
        self.clusters = to_keep

    def _check_if_message_outdated(self, message: GenericMessageType, cleanup: bool = True) -> bool:
        """Returns whether a message is outdated or not. Will also refresh the internal check timestamp."""
        self.latest_refresh_timestamp = max(message.encounter_time, self.latest_refresh_timestamp)
        outdated = False
        if self.latest_refresh_timestamp:
            min_offset = self.latest_refresh_timestamp - message.encounter_time
            if min_offset > self.max_history_offset:
                # there's no way this message is useful if we get here, since it's so old
                outdated = True
            if cleanup:
                self.cleanup_clusters(self.latest_refresh_timestamp)
        return outdated

    def add_messages(
            self,
            messages: MessagesArrayType,
            cleanup: bool = True,
            current_timestamp: typing.Optional[TimestampType] = None,  # will use internal latest if None
    ):
        """Dispatches the provided messages to the correct internal 'add' function based on type."""
        if current_timestamp is not None:
            self.latest_refresh_timestamp = max(current_timestamp, self.latest_refresh_timestamp)
        for message in messages:
            assert isinstance(message, (EncounterMessage, UpdateMessage, list, dict))
            if isinstance(message, EncounterMessage):
                self._add_encounter_message(message, cleanup=False)
            elif isinstance(message, UpdateMessage):
                self._add_update_message(message, cleanup=False)
            elif isinstance(message, (list, dict)) and message:
                self._add_update_message_batch(message, cleanup=False)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_encounter_message(self, message: EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        return NotImplementedError

    def _add_encounter_message_batch(self, messages: typing.List[EncounterMessage], cleanup: bool = True):
        """Fits a batch of encounter messages to existing clusters, and forward the remaining to non-batch impl."""
        # by default, just forward everything to the default impl
        # (only some clustering algos will be advantaged by having a custom impl here)
        for m in messages:
            self._add_encounter_message(m, cleanup=False)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def _add_update_message(self, message: UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        return NotImplementedError

    def _add_update_message_batch(self, messages: UpdateMessageBatchType, cleanup: bool = True):
        """Fits a batch of update messages to existing clusters, and forwards the remaining to non-batch impl."""
        # if no faster implementation is provided, just loop over messages individually
        if isinstance(messages, list):
            for msg in messages:
                self._add_update_message(msg)
        elif isinstance(messages, dict):
            for timestamp, msgs in messages.items():
                return self._add_update_message_batch(msgs)

    def set_current_timestamp(self, timestamp: TimestampType):
        """Sets the timestamp used internally to invalidate outdated messages/clusters."""
        assert timestamp >= self.latest_refresh_timestamp, "how could we have received future messages?"
        self.latest_refresh_timestamp = timestamp

    def get_embeddings_array(
            self,
            cleanup: bool = False,
            current_timestamp: typing.Optional[TimestampType] = None,  # will use internal latest if None
    ) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        if not self.generate_backw_compat_embeddings or not self.generate_embeddings_by_timestamp:
            raise NotImplementedError
        if current_timestamp is not None:
            self.latest_refresh_timestamp = max(current_timestamp, self.latest_refresh_timestamp)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)
        # note: we start the sequence with the OLDEST encounters, and move forward in time
        output = []
        target_timestamp = self.latest_refresh_timestamp - self.max_history_offset
        while target_timestamp <= self.latest_refresh_timestamp:
            for cluster in self.clusters:
                embed = cluster.get_cluster_embedding(
                    current_timestamp=target_timestamp,
                    include_cluster_id=True,
                    old_compat_mode=True,
                )
                if embed is not None:
                    output.append([*embed, (self.latest_refresh_timestamp - target_timestamp).days])
            target_timestamp += datetime.timedelta(days=1)
        return np.asarray(output)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        raise NotImplementedError

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
        raise NotImplementedError

    def _get_average_homogeneity(self) -> float:
        """Returns the average homogeneity score across all encountered users."""
        return float(np.mean(list(self._get_homogeneity_scores().values())))

    def get_cluster_count(self) -> int:
        """Returns the active cluster count in this object."""
        return len(self.clusters)

    def get_encounters_cluster_mapping(self) -> typing.List[typing.Tuple[EncounterMessage, ClusterIDType]]:
        """Returns a flattened list of encounters mapped to their cluster ids."""
        raise NotImplementedError

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
        raise NotImplementedError

    def __repr__(self):
        """For pretty printing."""
        return f"{type(self)}: nb_clusters={self.get_cluster_count()}"


def get_cluster_manager_type(algo_name: str):
    """Returns the type of cluster manager to instantiate based on algo name."""
    assert algo_name in ["blind", "naive", "perfect", "simple", "gaen"], \
        f"invalid clustering algo name: {algo_name}"
    if algo_name == "blind":
        import covid19sim.inference.clustering.blind
        return covid19sim.inference.clustering.blind.BlindClusterManager
    if algo_name == "naive":
        raise NotImplementedError("naive clustering algo deprecated since GAEN refactoring")
    if algo_name == "perfect":
        import covid19sim.inference.clustering.perfect
        return covid19sim.inference.clustering.perfect.PerfectClusterManager
    if algo_name == "simple":
        raise NotImplementedError("simple clustering algo deprecated since GAEN refactoring")
    if algo_name == "gaen":
        import covid19sim.inference.clustering.gaen
        return covid19sim.inference.clustering.gaen.GAENClusterManager
