import collections
import dataclasses
import numpy as np
import typing

from covid19sim.frozen.message_utils import EncounterMessage, GenericMessageType, UpdateMessage, \
    TimestampType, TimeOffsetType, RealUserIDType, RiskLevelType

ClusterIDType = int
MessagesArrayType = typing.Union[typing.Iterable[GenericMessageType],
                                 typing.Iterable[typing.List[GenericMessageType]]]


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

    messages: typing.List[EncounterMessage] = dataclasses.field(default_factory=list)
    """List of encounter messages aggregated into this cluster (in added order)."""
    # note: messages above might have been updated from their original state!

    ##########################################
    # private variables (for debugging only!)

    _real_encounter_uids: typing.List[RealUserIDType] = dataclasses.field(default_factory=list)
    """Real Unique Identifiers (UIDs) of the clustered user(s)."""

    _real_encounter_times: typing.List[TimestampType] = dataclasses.field(default_factory=list)
    """Real timestamp of the clustered encounter(s)."""

    _unclustered_messages: typing.List[GenericMessageType] = dataclasses.field(default_factory=list)
    """List of all messages (encounter+update) messages that were used to update this cluster."""

    def _is_homogenous(self) -> bool:
        """Returns whether this cluster is truly homogenous (i.e. tied to one user) or not."""
        return len(np.unique([m._sender_uid for m in self._unclustered_messages])) <= 1

    @staticmethod
    def create_cluster_from_message(
            message: GenericMessageType,
            cluster_id: typing.Optional[ClusterIDType] = None,  # unused by this base implementation
    ):
        """Creates and returns a new cluster based on a single encounter message."""
        raise NotImplementedError

    def fit_encounter_message(self, message: EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        raise NotImplementedError

    def fit_update_message(self, update_message: UpdateMessage):
        """Updates an encounter in the current cluster given a new update message."""
        raise NotImplementedError

    def get_cluster_embedding(self, include_cluster_id: bool) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        raise NotImplementedError

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event) for m in self.messages])


class ClusterManagerBase:
    """Manages message cluster creation and updates.

    This base class implements common utility functions used by other clustering algos.
    """

    clusters: typing.List[ClusterBase]
    latest_refresh_timestamp: TimestampType
    max_history_ticks_offset: TimeOffsetType
    add_orphan_updates_as_clusters: bool
    generate_embeddings_by_timestamp: bool

    def __init__(
            self,
            max_history_ticks_offset: TimeOffsetType = 24 * 60 * 60 * 14,  # one tick per second, 14 days
            add_orphan_updates_as_clusters: bool = False,
            generate_embeddings_by_timestamp: bool = True,
    ):
        self.clusters = []
        self.latest_refresh_timestamp = TimestampType(0)
        self.max_history_ticks_offset = max_history_ticks_offset
        self.add_orphan_updates_as_clusters = add_orphan_updates_as_clusters
        self.generate_embeddings_by_timestamp = generate_embeddings_by_timestamp

    def cleanup_clusters(self, current_timestamp: TimestampType):
        """Gets rid of clusters that are too old given the current timestamp."""
        to_keep = []
        for cluster in self.clusters:
            update_offset = int(current_timestamp) - int(cluster.latest_update_time)
            if update_offset <= self.max_history_ticks_offset:
                to_keep.append(cluster)
        self.clusters = to_keep

    def _check_if_message_outdated(self, message: GenericMessageType, cleanup: bool = True) -> bool:
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
            if isinstance(message, EncounterMessage):
                self._add_encounter_message(message, cleanup=False)
            elif isinstance(message, UpdateMessage):
                self._add_update_message(message, cleanup=False)
            elif isinstance(message, list):
                if message:
                    assert all([isinstance(m, EncounterMessage) for m in message]) or \
                        all([isinstance(m, UpdateMessage) for m in message]), \
                        "batched messages should all be the same type"
                    if isinstance(message[0], EncounterMessage):
                        self._add_encounter_message_batch(message, cleanup=False)
                    else:
                        self._add_update_message_batch(message, cleanup=False)
            else:
                ValueError("unexpected message type")
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

    def _add_update_message_batch(self, messages: typing.List[UpdateMessage], cleanup: bool = True):
        """Fits a batch of update messages to existing clusters, and forwards the remaining to non-batch impl."""
        # by default, just forward everything to the default impl
        # (only some clustering algos will be advantaged by having a custom impl here)
        for m in messages:
            self._add_update_message(m, cleanup=False)
        if cleanup:
            self.cleanup_clusters(self.latest_refresh_timestamp)

    def get_embeddings_array(self) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        if self.generate_embeddings_by_timestamp:
            cluster_embeds = collections.defaultdict(list)
            for cluster in self.clusters:
                embed = cluster.get_cluster_embedding(include_cluster_id=True)
                for msg in cluster.messages:
                    cluster_embeds[msg.encounter_time].append(embed)
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
                for msg in cluster.messages:
                    cluster_flags[msg.encounter_time].append(flags)
            flat_output = []
            for timestamp in sorted(cluster_flags.keys()):
                flat_output.extend(cluster_flags[timestamp])
            return np.asarray(flat_output)
        else:
            return np.asarray([c._get_cluster_exposition_flag() for c in self.clusters], dtype=np.uint8)
