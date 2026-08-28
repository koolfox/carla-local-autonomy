export function fieldValue(event: Event): string {
  return (event.currentTarget as HTMLInputElement | HTMLSelectElement).value;
}

export function fieldNumber(event: Event): number {
  return Number(fieldValue(event));
}

export function fieldChecked(event: Event): boolean {
  return (event.currentTarget as HTMLInputElement).checked;
}
