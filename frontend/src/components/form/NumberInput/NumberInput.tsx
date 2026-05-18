import { Label, TextInput as FBTextInput, HelperText } from "flowbite-react";

interface NumberInputProps {
  name: string;
  value: number;
  onChange: (s: number) => void;
  label: string;
  placeholder?: string;
  error?: string;
}

function NumberInput({name, value, label, onChange, placeholder, error}: NumberInputProps) {
  return (
    <div>
      <div className="mb-2 block">
        <Label htmlFor={name} color={error ? 'failure' : 'default'}>{label}</Label>
      </div>
      <FBTextInput
        id={name}
        type="number"
        color={error ? 'failure' : 'gray'}
        onChange={(e) => onChange(parseInt(e.target.value))}
        value={value}
        placeholder={placeholder ?? ''}
        shadow
      />
      {error && <HelperText color="failure">{error}</HelperText>}
    </div>
  )
}

export default NumberInput