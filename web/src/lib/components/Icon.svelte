<script lang="ts" module>
  // One 24-unit grid, 1.6 stroke, round joins: every glyph in the console shares this drawing.
  const paths = {
    experiment: 'M9 3h6M10 3v6l-5.2 9A2 2 0 0 0 6.5 21h11a2 2 0 0 0 1.7-3L14 9V3M7.5 15h9',
    scene: 'M3 19 9 9l4 6 3-4 5 8H3ZM16 6.5a1.5 1.5 0 1 0 0-.01',
    vehicle: 'M4 16v-3.5L6 7.8A2 2 0 0 1 7.8 6.5h8.4a2 2 0 0 1 1.8 1.3l2 4.7V16M3.5 16h17v2.5h-17zM6.5 18.5V20M17.5 18.5V20M4.5 12.5h15M7.5 14.3h.01M16.5 14.3h.01',
    vision: 'M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12ZM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    research: 'M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3ZM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6',
    recordings: 'M4 5.5h16v13H4zM4 9.5h16M8 5.5l-1.5 4M13 5.5l-1.5 4M18 5.5l-1.5 4M10.5 12.5v4l3.5-2z',
    system: 'M7 7h10v10H7zM10 3.5V7M14 3.5V7M10 17v3.5M14 17v3.5M3.5 10H7M3.5 14H7M17 10h3.5M17 14h3.5',
    'chevron-left': 'm14.5 6-6 6 6 6',
    'chevron-right': 'm9.5 6 6 6-6 6',
    'panel-right': 'M4 5h16v14H4zM14.5 5v14',
    close: 'M6.5 6.5l11 11M17.5 6.5l-11 11',
    retry: 'M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4',
    play: 'M7.5 5.5v13l11-6.5z',
    stop: 'M7 7h10v10H7z',
    brake: 'M8.3 3.5h7.4l4.8 4.8v7.4l-4.8 4.8H8.3l-4.8-4.8V8.3zM8 12h8',
    check: 'm5 12.5 4.5 4.5L19 7.5',
    sliders: 'M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4',
    'arrow-left': 'M19 12H5M11 6l-6 6 6 6',
    'arrow-right': 'M5 12h14M13 6l6 6-6 6',
    'arrow-up': 'M12 19V5M6 11l6-6 6 6',
    manual: 'M12 20.5a8.5 8.5 0 1 0 0-17 8.5 8.5 0 0 0 0 17ZM12 14.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5ZM3.8 10.5h5.8M14.4 10.5h5.8M12 14.5v6',
    autopilot: 'M8.5 2.5h7v19h-7zM12 7h.01M12 12h.01M12 17h.01M5 5.5h3.5M15.5 5.5H19M5 10.5h3.5M15.5 10.5H19',
    behavior: 'M6 20.5v-6a4 4 0 0 1 4-4h8M15 7.5l3 3-3 3M6 3.5v3',
    imitation: 'M8.5 8.5h11v11h-11zM5.5 15.5h-1v-11h11v1',
    voxel: 'M12 2.8 20 7.4v9.2l-8 4.6-8-4.6V7.4zM4 7.4l8 4.6 8-4.6M12 12v9.2',
    model: 'M5 6.5 12 3l7 3.5-7 3.5zM5 12l7 3.5 7-3.5M5 17.5 12 21l7-3.5',
    alert: 'M12 4 2.8 19.5h18.4zM12 10v4.5M12 17.2h.01'
  } as const;

  export type IconName = keyof typeof paths;
</script>

<script lang="ts">
  export let name: IconName;
  export let size = 16;
</script>

<svg
  class="icon"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.6"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
  focusable="false"
><path d={paths[name]} /></svg>
