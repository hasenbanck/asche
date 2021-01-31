/// Errors that the allocator can throw.
#[derive(Debug)]
pub enum AllocatorError {
    /// General out of memory error.
    OutOfMemory,
    /// Failed to map the memory.
    FailedToMap,
    /// No compatible memory type was found.
    NoCompatibleMemoryTypeFound,
    /// Invalid AllocationDescriptor.
    InvalidAllocationDescriptor,
    /// An allocator implementation error.
    Internal(&'static str),
}

impl std::fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AllocatorError::OutOfMemory => {
                write!(f, "out of memory")
            }
            AllocatorError::FailedToMap => {
                write!(f, "failed to map memory")
            }
            AllocatorError::NoCompatibleMemoryTypeFound => {
                write!(f, "no compatible memory type available")
            }
            AllocatorError::InvalidAllocationDescriptor => {
                write!(f, "invalid AllocationDescriptor")
            }
            AllocatorError::Internal(message) => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for AllocatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
