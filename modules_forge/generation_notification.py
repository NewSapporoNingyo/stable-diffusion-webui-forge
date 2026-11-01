import sys


def notify_generation_finished():
    if sys.platform != "win32":
        return

    from modules import errors, shared

    if not shared.opts.notification_toast or shared.state.interrupted or shared.state.stopping_generation:
        return

    try:
        from windows_toasts import Toast, WindowsToaster

        toast = Toast()
        toast.text_fields = ["WebUI Generation Finished"]
        WindowsToaster("Stable Diffusion WebUI Forge").show_toast(toast)
    except Exception as e:  # noqa: BLE001
        errors.display_once(e, "sending generation-finished Windows notification")
