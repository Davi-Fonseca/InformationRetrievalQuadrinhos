import { Label, TextInput as FBTextInput, HelperText } from "flowbite-react";
import { useEffect, useRef } from "react";

interface TextInputProps {
  name: string;
  value: string;
  onChange: (s: string) => void;
  label: string;
  placeholder?: string;
  error?: string;
  focus?: boolean;
  type?: string
}

function TextInput({name, value, label, onChange, placeholder, error, focus, type}: TextInputProps) {
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (focus) ref.current?.focus()
  }, [focus])

  return (
    <div>
      <div className="mb-2 block">
        <Label htmlFor={name} color={error ? 'failure' : 'default'}>{label}</Label>
      </div>
      <FBTextInput
        ref={ref}
        id={name}
        type={type ?? "text"}
        onChange={(e) => onChange(e.target.value)}
        value={value}
        placeholder={placeholder ?? ''}
        color={error ? 'failure' : 'gray'}
        shadow
      />
      {error && <HelperText color="failure">{error}</HelperText>}
    </div>
  )
}

export default TextInput