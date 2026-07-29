import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { chromium } from "playwright-core";

import {
  fontSizeForScale,
  type FontScale,
} from "../lib/useAppearance";


const css = readFileSync(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);
const scales: FontScale[] = ["xs", "sm", "md", "lg", "xl"];


test(
  "five font-size levels increase computed sizes across primary UI",
  { timeout: 30_000 },
  async () => {
    const browser = await chromium.launch({
      channel: "msedge",
      headless: true,
    });
    try {
      const page = await browser.newPage();
      await page.setContent(`
        <style>${css}</style>
        <aside class="sidebar">
          <button id="nav" class="nav-item">新建任务</button>
        </aside>
        <main>
          <header class="page-header"><h1 id="title">设置</h1></header>
          <button id="button" class="button">开始生成</button>
          <label class="field"><span id="field">识别语言</span></label>
        </main>
      `);

      const samples: Record<string, number[]> = {
        root: [],
        nav: [],
        title: [],
        button: [],
        field: [],
      };
      for (const scale of scales) {
        await page.evaluate((baseFontSize) => {
          document.documentElement.style.setProperty(
            "--base-font-size",
            baseFontSize,
          );
        }, fontSizeForScale(scale));
        const computed = await page.evaluate(() => {
          return {
            root: Number.parseFloat(
              getComputedStyle(document.documentElement).fontSize,
            ),
            nav: Number.parseFloat(
              getComputedStyle(document.querySelector("#nav")!).fontSize,
            ),
            title: Number.parseFloat(
              getComputedStyle(document.querySelector("#title")!).fontSize,
            ),
            button: Number.parseFloat(
              getComputedStyle(document.querySelector("#button")!).fontSize,
            ),
            field: Number.parseFloat(
              getComputedStyle(document.querySelector("#field")!).fontSize,
            ),
          };
        });
        for (const key of Object.keys(samples)) {
          samples[key]!.push(computed[key as keyof typeof computed]);
        }
      }

      for (const [name, values] of Object.entries(samples)) {
        assert.equal(values.length, scales.length);
        for (let index = 1; index < values.length; index += 1) {
          assert.ok(
            values[index]! > values[index - 1]!,
            `${name} should grow from ${scales[index - 1]} to ${scales[index]}`,
          );
        }
      }
    } finally {
      await browser.close();
    }
  },
);
