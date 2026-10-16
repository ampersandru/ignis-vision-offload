import type { ImageBackground } from "../api/images.ts";
import type { SessionDraft } from "./sessions.ts";
import { AUTO_SIZE } from "./studio.ts";

// Starting points for the image view. A new session offers pictures of the
// kinds Qwen-Image-2.1 is good at — photographs, illustration and 3D, text set
// in the picture (signs, slides, infographics, posters, comics), reference
// sheets, product shots, transparent stickers — each written the way the model
// reads best: concrete, the words to render in quotes. Picking one fills the
// prompt, the shape and the background. A step with inputs and no prompt yet
// offers edit ideas instead, two-picture ones when there are two inputs.

export type PictureExample = {
  id: string;
  /** What kind of picture it is: the row's label. */
  kind: string;
  title: string;
  /** One of the model's presets (spec image-model/05 §Sizes). */
  size: string;
  background?: Extract<ImageBackground, "transparent">;
  prompt: string;
};

export const PICTURE_EXAMPLES: PictureExample[] = [
  {
    id: "portrait",
    kind: "Photograph",
    title: "Portrait in window light",
    size: "1792x2400",
    prompt:
      "A close-up portrait of an elderly fisherman with a weathered face, a grey wool cap and a thick knitted sweater, soft window light from the left, shallow depth of field, fine film grain, muted teal and amber tones, shot on an 85 mm lens.",
  },
  {
    id: "lighthouse",
    kind: "Landscape",
    title: "Lighthouse at dawn",
    size: "2752x1536",
    prompt:
      "At dawn a thin mist veils the sea. An old stone lighthouse stands at the cliff's edge, its lamp still glowing; waves break white on dark rocks below, the horizon glows soft pink, wide-angle, photorealistic.",
  },
  {
    id: "chalkboard",
    kind: "Text in the picture",
    title: "A café's chalkboard sign",
    size: "2400x1792",
    prompt:
      'A small café entrance at night with a chalkboard sign reading "Ignis Coffee — espresso €1.20", hand-lettered in white chalk with a little flame doodle; warm light spills from the window onto wet cobblestones.',
  },
  {
    id: "slide",
    kind: "Slide",
    title: "Quarterly results",
    size: "2752x1536",
    prompt:
      'A modern presentation slide on a deep navy gradient. At the top, the title "Q3 Results" in bold white sans-serif. Below, three panels side by side with large numbers "+24%", "1.2M" and "99.9%" over small grey labels "Revenue", "Users" and "Uptime"; a thin orange line chart runs along the bottom.',
  },
  {
    id: "infographic",
    kind: "Infographic",
    title: "How an espresso is made",
    size: "1696x2528",
    prompt:
      'A clean vertical infographic titled "How an espresso is made" on a pale sage background: five numbered steps from top to bottom — "Grind", "Dose", "Tamp", "Extract", "Serve" — each with a flat line icon, a one-line caption, and its time or temperature in bold; deep brown accents, modern sans-serif type, generous margins.',
  },
  {
    id: "poster",
    kind: "Poster",
    title: "Vintage travel poster",
    size: "1696x2528",
    prompt:
      'A vintage travel poster for the Dolomites: jagged peaks at sunset in flat screen-print colours, a red cable car crossing the valley, the word "DOLOMITI" in large art-deco letters at the top and "Ferrovie delle Alpi" in small capitals at the bottom, slightly worn paper texture.',
  },
  {
    id: "grid",
    kind: "Photo grid",
    title: "A healthy day, in twelve frames",
    size: "1792x2400",
    prompt:
      'A 3 × 4 grid of twelve photographs telling the story of "A healthy day", from a sunrise run and a breakfast bowl to a bike commute, a salad lunch, a yoga class and an early night with a book; consistent soft natural light, each frame with a small white caption in the corner.',
  },
  {
    id: "sheet",
    kind: "Reference sheet",
    title: "Character turnaround",
    size: "2752x1536",
    prompt:
      "A character reference sheet of a young explorer in a green raincoat, yellow boots and a battered leather satchel: front, side and back views in a row on a plain light grey background, a colour palette strip below, and three facial expressions in a corner; clean anime line art with flat colours.",
  },
  {
    id: "comic",
    kind: "Comic",
    title: "Four panels",
    size: "1792x2400",
    prompt:
      'A four-panel comic strip in a clean webcomic style: a small robot tries to water a cactus, gets pricked, puts on oven mitts, and finally waters it proudly. Speech bubbles: "Hello, plant!", "Ouch!", "Safety first." and "Thriving!".',
  },
  {
    id: "product",
    kind: "Product shot",
    title: "Mug on travertine",
    size: "2048x2048",
    prompt:
      "A product photograph of a matte black ceramic coffee mug on a slab of travertine, a single hard light from the right casting a long crisp shadow, a sprig of rosemary beside it, minimal studio setting, very high detail.",
  },
  {
    id: "sticker",
    kind: "Sticker",
    title: "A fox with coffee, cut out",
    size: "2048x2048",
    background: "transparent",
    prompt: "A cute cartoon fox sticker with a thick white outline, sitting and holding a steaming cup of coffee in both paws, flat colours with soft shading.",
  },
  {
    id: "watercolour",
    kind: "Illustration",
    title: "A Venetian canal in watercolour",
    size: "2528x1696",
    prompt:
      "A watercolour illustration of a Venetian canal at noon: pastel houses with laundry lines, a gondola in the foreground, loose brushstrokes and soft bleeding edges, the white of the paper showing through.",
  },
  {
    id: "diorama",
    kind: "3D",
    title: "A ramen shop diorama",
    size: "2048x2048",
    prompt:
      "A cute isometric 3D diorama of a tiny ramen shop at night: glowing paper lanterns, steam rising from the bowls, a cat asleep on the counter, soft clay-like materials, gentle pastel lighting.",
  },
  {
    id: "botanical",
    kind: "Scientific plate",
    title: "A botanical plate",
    size: "1792x2400",
    prompt:
      'A vintage botanical plate of the common fig: a branch with leaves, a whole fruit and a cross-section, finely engraved and hand-coloured, labelled in neat italic serif "Ficus carica" with small numbered annotations, on aged cream paper.',
  },
];

/** What picking an example puts in the draft: its prompt, its shape if the server serves it (else auto), its background. */
export function exampleDraft(example: PictureExample, sizes: string[]): Partial<SessionDraft> {
  return {
    prompt: example.prompt,
    size: sizes.includes(example.size) ? example.size : AUTO_SIZE,
    background: example.background ?? "auto",
  };
}

export type EditIdea = {
  id: string;
  label: string;
  prompt: string;
  background?: Extract<ImageBackground, "transparent">;
  /** The inputs it needs. */
  inputs: 1 | 2;
};

export const EDIT_IDEAS: EditIdea[] = [
  { id: "watercolour", label: "Watercolour", prompt: "Turn image 1 into a loose watercolour painting, keeping its composition.", inputs: 1 },
  { id: "golden", label: "Golden hour", prompt: "Relight image 1 at golden hour: a warm, low sun from the left and long soft shadows.", inputs: 1 },
  { id: "cutout", label: "Cut out", prompt: "Extract the main subject of image 1 onto a transparent background, edges clean.", background: "transparent", inputs: 1 },
  { id: "winter", label: "Winter", prompt: "Make it winter in image 1: snow on the ground and the roofs, a pale cold light.", inputs: 1 },
  { id: "clay", label: "3D clay figure", prompt: "Turn the subject of image 1 into a cute 3D clay figure on a plain pastel background.", inputs: 1 },
  { id: "angle", label: "New angle", prompt: "Show the subject of image 1 from a three-quarter view from slightly above, same lighting and style.", inputs: 1 },
  { id: "tidy", label: "Remove the clutter", prompt: "Remove the people and the clutter from the background of image 1, keeping everything else as it is.", inputs: 1 },
  { id: "scene", label: "Into the scene", prompt: "Put the person from image 1 into the scene of image 2, matching its light and perspective.", inputs: 2 },
  { id: "outfit", label: "Swap the outfit", prompt: "Dress the person in image 1 in the outfit from image 2, keeping their face and pose.", inputs: 2 },
];

/** The edit ideas a step with `inputs` inputs can use. */
export const ideasFor = (inputs: number) => (inputs === 0 ? [] : EDIT_IDEAS.filter((idea) => idea.inputs <= inputs));
