from numpy cimport uint32_t
from covid19sim.frozen.message_utils cimport (
    GenericMessageCythonType,
    EncounterMessage,
    UpdateMessage,
    TimestampCythonType,
    TimeOffsetCythonType,
    RealUserIDCythonType,
    RiskLevelCythonType,
    TimestampDefault
    )
from numpy cimport ndarray

ctypedef uint32_t ClusterIDCythonType

ctypedef fused UpdateMessageBatchCythonType:
    list
    dict

cdef class ClusterBase:
    """An encounter message cluster."""
    cdef public:
        ClusterIDCythonType cluster_id
        """Unique Identifier (UID) of the cluster."""

        RiskLevelCythonType risk_level
        """Quantified risk level of the cluster."""

        TimestampCythonType first_update_time
        """Cluster creation timestamp (i.e. timestamp of first encounter)."""

        TimestampCythonType latest_update_time
        """Latest cluster update timestamp (i.e. timestamp of latest encounter)."""

        ###########################################
        # private variables (for debugging only!) #
        ###########################################
        set _real_encounter_uids
        """Real Unique Identifiers (UIDs) of the clustered user(s)."""

        set _real_encounter_times
        """Real timestamp of the clustered encounter(s)."""

cdef class ClusterManagerBase:
    """Manages message cluster creation and updates.

    This base class implements common utility functions used by other clustering algos.
    """
    cdef public:
        list clusters
        TimestampCythonType latest_refresh_timestamp
        TimeOffsetCythonType max_history_offset
        bint add_orphan_updates_as_clusters
        bint generate_embeddings_by_timestamp
        bint generate_backw_compat_embeddings
        ClusterIDCythonType max_cluster_id

        # @cython.locals(to_keep = cython.list,# cluster = ClusterBase,
        # update_offset = TimeOffsetCythonType, ind = uint32_t)
        cdef void cleanup_clusters(self, TimestampCythonType current_timestamp)
        cdef bint _check_if_message_outdated(
            self,
            GenericMessageCythonType message,
            bint cleanup = *
            )
        cpdef void add_messages(
            self,
            list messages,
            bint cleanup = *,
            TimestampCythonType current_timestamp = *,  # will use internal latest if None
        )
        cdef void _add_encounter_message_batch(
            self,
            list messages,
            bint cleanup = *
        )
        # cdef void _add_update_message_batch(
        # self, 
        # UpdateMessageBatchCythonType messages, 
        # bint cleanup = *
        # )
        cpdef void set_current_timestamp(self, TimestampCythonType timestamp) except *
        cpdef ndarray get_embeddings_array(
            self,
            bint cleanup = *,
            TimestampCythonType current_timestamp = *,  # will use internal latest if None
        )
