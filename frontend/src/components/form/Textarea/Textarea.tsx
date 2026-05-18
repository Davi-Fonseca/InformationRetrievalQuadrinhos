import { Label, Textarea as FBTextarea } from "flowbite-react";

interface TextareaProps {
  name: string;
  value: string;
  onChange: (s: string) => void;
  label: string;
  placeholder?: string;
  required?: boolean;
  rows?: number;
}

function Textarea({name, value, label, onChange, placeholder, required, rows}: TextareaProps) {
  return (
    <div>
      <div className="mb-2 block">
        <Label htmlFor={name}>{label}</Label>
      </div>
      <FBTextarea
        id={name}
        onChange={(e) => onChange(e.target.value)}
        value={value}
        placeholder={placeholder ?? ''}
        required={required ?? false}
        shadow
        rows={rows ?? 4}
      />
    </div>
  )
}

export default Textarea