// CARLA supplies RGB paint values, not manufacturer paint names.
// Keep the original value for the API; these are approximate display names.
const paints = [
  ['Black', '⚫', 0, 0, 0],
  ['Charcoal', '⚫', 55, 55, 55],
  ['Gray', '🩶', 128, 128, 128],
  ['Silver', '⚪', 192, 192, 192],
  ['White', '⚪', 255, 255, 255],
  ['Red', '🔴', 220, 25, 25],
  ['Burgundy', '🔴', 100, 15, 35],
  ['Orange', '🟠', 245, 130, 25],
  ['Yellow', '🟡', 245, 220, 30],
  ['Gold', '🟡', 185, 145, 45],
  ['Green', '🟢', 40, 150, 55],
  ['Dark green', '🟢', 15, 65, 30],
  ['Teal', '🟢', 20, 130, 130],
  ['Blue', '🔵', 30, 90, 220],
  ['Navy', '🔵', 15, 30, 80],
  ['Light blue', '🔵', 130, 195, 230],
  ['Purple', '🟣', 130, 50, 170],
  ['Pink', '🩷', 235, 145, 180],
  ['Brown', '🟤', 110, 65, 35],
  ['Beige', '🟤', 210, 190, 150]
] as const;

export function paintLabel(value: string): string {
  const raw = value.trim();
  const rgb = /^#[\da-f]{6}$/i.test(raw)
    ? [1, 3, 5].map((offset) => parseInt(raw.slice(offset, offset + 2), 16))
    : /^\d+\s*,\s*\d+\s*,\s*\d+$/.test(raw)
      ? raw.split(',').map(Number) : [];
  if (rgb.length !== 3 || rgb.some((channel) => channel < 0 || channel > 255)) {
    return '◌ Custom paint';
  }
  const nearest = paints.reduce((best, paint) => {
    const distance = (candidate: typeof paints[number]) =>
      rgb.reduce((sum, channel, index) => sum + (channel - (candidate[index + 2] as number)) ** 2, 0);
    return distance(paint) < distance(best) ? paint : best;
  });
  return `${nearest[1]} ${nearest[0]}`;
}
