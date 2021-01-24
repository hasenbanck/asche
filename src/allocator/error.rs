/// Errors that the allocator module can throw.
#[derive(Debug)]
pub enum AllocationError {
    /// General out of memory error.
    OutOfMemory,
    /// Failed to map the memory.
    FailedToMap,
    /// No compatible memory type was found.
    NoCompatibleMemoryTypeFound,
    /// Invalid AllocationDescriptor.
    InvalidAllocationCreateDesc,
    /// An allocator implementation error.
    Internal(&'static str),
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AllocationError::OutOfMemory => {
                write!(f, "out of memory")
            }
            AllocationError::FailedToMap => {
                write!(f, "failed to map memory")
            }
            AllocationError::NoCompatibleMemoryTypeFound => {
                write!(f, "no compatible memory type available")
            }
            AllocationError::InvalidAllocationCreateDesc => {
                write!(f, "invalid AllocationDescriptor")
            }
            AllocationError::Internal(message) => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for AllocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
