<script lang="ts">
  export let open = false;
  export let title: string;
  export let description = '';
  export let close: () => void;

  let dialog: HTMLDialogElement;

  $: if (dialog) {
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }

  function dismiss(event?: MouseEvent): void {
    if (event && event.target !== dialog) return;
    close();
  }
</script>

<dialog
  bind:this={dialog}
  class="app-modal"
  aria-label={title}
  onclose={close}
  onclick={dismiss}
>
  <div class="modal-surface">
    <header class="modal-header">
      <div>
        <h2>{title}</h2>
        {#if description}<p>{description}</p>{/if}
      </div>
      <button type="button" class="modal-close" aria-label={`Close ${title}`} onclick={close}>×</button>
    </header>
    <div class="modal-body">
      <slot></slot>
    </div>
  </div>
</dialog>
