"use client";

import { ChevronDown } from "lucide-react";
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";


export interface SelectOption {
  value: string;
  label: string;
}

interface CustomSelectProps {
  value: string;
  options: SelectOption[];
  disabled?: boolean;
  onChange: (value: string) => void;
}


export function CustomSelect({
  value,
  options,
  disabled,
  onChange,
}: CustomSelectProps) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();
  const triggerId = `${listboxId}-trigger`;

  const selectedOption = options.find((option) => option.value === value);
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );

  const openMenu = useCallback(
    (index = selectedIndex) => {
      if (disabled || options.length === 0) {
        return;
      }
      setActiveIndex(Math.max(0, Math.min(index, options.length - 1)));
      setOpen(true);
    },
    [disabled, options.length, selectedIndex],
  );

  const selectIndex = useCallback(
    (index: number) => {
      const option = options[index];
      if (!option) {
        return;
      }
      onChange(option.value);
      setActiveIndex(index);
      setOpen(false);
    },
    [onChange, options],
  );

  const handleTriggerKeyDown = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
  ) => {
    if (disabled || options.length === 0) {
      return;
    }
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        if (!open) {
          openMenu(selectedIndex);
        } else {
          setActiveIndex((index) => Math.min(index + 1, options.length - 1));
        }
        break;
      case "ArrowUp":
        event.preventDefault();
        if (!open) {
          openMenu(selectedIndex);
        } else {
          setActiveIndex((index) => Math.max(index - 1, 0));
        }
        break;
      case "Home":
        event.preventDefault();
        openMenu(0);
        break;
      case "End":
        event.preventDefault();
        openMenu(options.length - 1);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        if (open) {
          selectIndex(activeIndex);
        } else {
          openMenu();
        }
        break;
      case "Escape":
        if (open) {
          event.preventDefault();
          setOpen(false);
        }
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  const handleOutsideClick = useCallback((event: MouseEvent) => {
    if (
      containerRef.current &&
      !containerRef.current.contains(event.target as Node)
    ) {
      setOpen(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      document.addEventListener("mousedown", handleOutsideClick);
      return () =>
        document.removeEventListener("mousedown", handleOutsideClick);
    }
  }, [open, handleOutsideClick]);

  return (
    <div
      ref={containerRef}
      className={`custom-select${open ? " is-open" : ""}${disabled ? " is-disabled" : ""}`}
    >
      <button
        id={triggerId}
        type="button"
        role="combobox"
        className="custom-select-trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-activedescendant={
          open ? `${listboxId}-option-${activeIndex}` : undefined
        }
        onClick={() => {
          if (open) {
            setOpen(false);
          } else {
            openMenu();
          }
        }}
        onKeyDown={handleTriggerKeyDown}
      >
        <span className="custom-select-value">
          {selectedOption?.label ?? "请选择"}
        </span>
        <ChevronDown size={15} className="custom-select-arrow" aria-hidden="true" />
      </button>

      {open ? (
        <ul
          id={listboxId}
          className="custom-select-menu"
          role="listbox"
          aria-labelledby={triggerId}
        >
          {options.map((option, index) => (
            <li
              id={`${listboxId}-option-${index}`}
              key={option.value}
              role="option"
              aria-selected={option.value === value}
              className={`custom-select-option${option.value === value ? " is-selected" : ""}${index === activeIndex ? " is-active" : ""}`}
              onMouseEnter={() => setActiveIndex(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => selectIndex(index)}
            >
              {option.label}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
