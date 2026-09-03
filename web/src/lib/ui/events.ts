export function fieldValue(event: Event): string {
  return (event.currentTarget as HTMLInputElement | HTMLSelectElement).value;
}

export function fieldNumber(event: Event): number {
  const value = fieldValue(event).trim();
  return value === '' ? Number.NaN : Number(value);
}

export function fieldChecked(event: Event): boolean {
  return (event.currentTarget as HTMLInputElement).checked;
}
