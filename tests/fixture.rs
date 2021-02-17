#[cfg(target_family = "unix")]
use winit::platform::unix::EventLoopExtUnix;
#[cfg(target_family = "windows")]
use winit::platform::windows::EventLoopExtWindows;

#[cfg(feature = "tracing")]
pub fn initialize_logging() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        use tracing_subscriber::filter::EnvFilter;

        let filter = EnvFilter::from_default_env().add_directive("asche=WARN".parse().unwrap());
        tracing_subscriber::fmt().with_env_filter(filter).init();
    });
}

pub struct TestContext {
    pub window: winit::window::Window,
}

impl Default for TestContext {
    fn default() -> Self {
        #[cfg(feature = "tracing")]
        initialize_logging();

        let event_loop: winit::event_loop::EventLoop<()> =
            winit::event_loop::EventLoop::new_any_thread();
        let window = winit::window::Window::new(&event_loop).unwrap();

        Self { window }
    }
}
