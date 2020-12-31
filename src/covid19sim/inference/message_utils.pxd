import cython
from numpy cimport int8_t, uint64_t

ctypedef uint64_t UIDCythonType  # should be at least 16 bytes for truly unique keys?
ctypedef object TimestampCythonType
ctypedef object TimeOffsetCythonType
ctypedef str RealUserIDCythonType
ctypedef int8_t RiskLevelCythonType

message_uid_bit_count = cython.declare(UIDCythonType)
risk_level_bit_count = cython.declare(RiskLevelCythonType)
message_uid_mask = cython.declare(UIDCythonType)
risk_level_mask = cython.declare(RiskLevelCythonType)
TimestampDefault = cython.declare(TimestampCythonType)

ctypedef fused GenericMessageCythonType:
    EncounterMessage
    UpdateMessage

cdef class UpdateMessage:
    cdef public:
        ######################
        # observed variables #
        ######################
        UIDCythonType uid

        RiskLevelCythonType old_risk_level
        """Previous quantified risk level of the updater."""

        RiskLevelCythonType new_risk_level
        """New quantified risk level of the updater."""

        TimestampCythonType encounter_time
        """Discretized encounter timestamp."""

        TimestampCythonType update_time
        """Update generation timestamp.""" # TODO: this might be a 1-31 rotating day id?

        ##############################################
        # Unobserved variables (for debugging only!) #
        ##############################################
        short _order_offset # 1 means that a direct contact had a new cause for update
        """Defines the 'order' distance of the original cause of the update."""

        RealUserIDCythonType _sender_uid
        """Real Unique Identifier (UID) of the encountered user."""

        RealUserIDCythonType _receiver_uid
        """Real Unique Identifier (UID) of the user receiving the message."""

        TimestampCythonType _real_encounter_time
        """Real encounter timestamp."""

        TimestampCythonType _real_update_time
        """Real update generation timestamp."""

        str _update_reason
        """Reason why this update message was sent (for debugging)."""

cdef class EncounterMessage:
    """Contains all the observed+unobserved data related to a user encounter message."""

    cdef public:
        ######################
        # observed variables #
        ######################
        UIDCythonType uid
        """Unique Identifier (UID) of the encountered user."""

        RiskLevelCythonType risk_level
        """Quantified risk level of the encountered user."""

        TimestampCythonType encounter_time
        """Discretized encounter timestamp."""

        ##############################################
        # Unobserved variables (for debugging only!) #
        ##############################################
        RealUserIDCythonType _sender_uid
        """Real Unique Identifier (UID) of the encountered user."""

        RealUserIDCythonType _receiver_uid
        """Real Unique Identifier (UID) of the user receiving the message."""

        TimestampCythonType _real_encounter_time
        """Real encounter timestamp."""

        int8_t _exposition_event
        """Flags whether this encounter corresponds to an exposition event for the receiver."""

        cython.list _applied_updates  # note: not used in clustering
        """List of update messages which have been applied to this encounter."""

@cython.locals(bits_left = UIDCythonType)
cpdef UIDCythonType create_new_uid(rng=*)

@cython.locals(rounded_timestamp = TimestampCythonType)
cdef EncounterMessage generate_encounter_message(
        cython.object sender,
        cython.object receiver,
        TimestampCythonType env_timestamp,
        cython.short minutes_granularity,
    )

cpdef UpdateMessage create_update_message(
        EncounterMessage encounter_message,
        RiskLevelCythonType new_risk_level,
        TimestampCythonType current_time,
        cython.short order_offset = *,
        cython.str update_reason = *,
    )

cpdef EncounterMessage create_encounter_from_update_message(
        UpdateMessage update_message,
)

cpdef EncounterMessage create_updated_encounter_with_message(
        EncounterMessage encounter_message,
        UpdateMessage update_message,
        cython.bint blind_update = *,
    )
