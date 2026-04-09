"use client";

import { useMemo, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  BarChart3,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  Trophy,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

type Label = "Set 1" | "Set 2" | "Set 3";
type PromptMap = Record<string, string>;

const prompts: PromptMap = {
  "0.mp4": "A person unboxing the latest iPhone 16 Pro in Desert Titanium on a wooden table, front view.",
  "1.mp4": "A close-up of a Tesla Cybertruck driving through a puddle, low angle.",
  "2.mp4": "A person opening a bottle of Coca-Cola with the new attached tethered cap, side view.",
  "3.mp4": "A person correctly using a French Press to brew coffee, countertop view.",
  "4.mp4": "A close-up of a Newton's Cradle in motion on a desk, macro view.",
  "5.mp4": "A person tying a Bowline knot with a thick rope, hands-only close-up.",
  "7.mp4": "A yellow New York City taxi driving through Times Square, street-level view.",
  "8.mp4": "A person walking across the Abbey Road zebra crossing, eye-level view.",
  "9.mp4": "A Mongolian horseman using an Uurga to catch a wild horse, wide landscape shot.",
  "10.mp4": "A samurai practicing with a nodachi.",
  "11.mp4": "The lifecycle of a Monarch butterfly: the chrysalis stage.",
  "12.mp4": "A chef preparing Hand-Pulled Lanzhou Ramen (Lamian).",
  "13.mp4": "A person playing a game of 'Jenga' and pulling out a middle block.",
  "14.mp4": "A chemist performing a 'Titration' until the exact 'End Point' is reached.",
  "15.mp4": "A person using a 'Self-Checkout' machine at a grocery store.",
  "16.mp4": "A close-up of a 'Vinyl Record Player' being started.",
  "17.mp4": "A person changing a flat tire on a car.",
  "18.mp4": "A dancer performing the 'Haka' with correct 'Pukana' facial expressions.",
  "19.mp4": "A person wearing a 'Hennin' headdress walking through a 15th-century court.",
  "20.mp4": "A close-up of a 'Mantis Shrimp' punching a glass tank.",
  "21.mp4": "A person making a peanut butter and jelly sandwich.",
  "22.mp4": "A person mailing a letter at a blue USPS mailbox.",
  "23.mp4": "A technician performing a 'Three-Point Bend' test on a carbon fiber sample.",
  "24.mp4": "A barista preparing a traditional Turkish coffee using a cezve in hot sand.",
  "25.mp4": "An archer demonstrating the Khatra technique upon releasing an arrow, slow-motion side view.",
  "26.mp4": "An artisan practicing Kintsugi on a broken ceramic bowl, soft-lit tabletop scene.",
};

const set1: PromptMap = {
  "0.mp4": "[00:00-00:03] Top-down shot of a pristine iPhone box on a rustic wooden table; hands slowly peel back the paper security tabs. [00:03-00:05] Front view as the lid is lifted, revealing the Desert Titanium finish catching the light. [00:05-00:08] Close-up of a hand lifting the phone out of the box, showing the side buttons and texture. SFX: The crisp sound of peeling adhesive followed by a soft \"thump\" of the lid hitting the table.",
  "1.mp4": "[00:00-00:03] Low-angle ground shot of a deep puddle; the reflection of the sky is broken as the stainless steel tire enters the frame. [00:03-00:05] Slow-motion tracking shot at wheel level as water sprays in a perfect arc against the geometric body. [00:05-00:08] Wide rear shot as the truck speeds away, water dripping from the angular tailgate. SFX: A deep, electric hum and a heavy splash of water.",
  "2.mp4": "[00:00-00:03] Side view of a hand gripping a Coke bottle; the thumb twists the new cap, which stays attached to the neck ring. [00:03-00:06] Close-up of the cap flipping back and clicking into place, showing it won't fall off. [00:06-00:08] Medium shot of the person taking a sip with the cap safely out of the way. SFX: The classic \"hiss\" of carbonation followed by a plastic click.",
  "3.mp4": "[00:00-00:03] Countertop view of hot water being poured over coarse coffee grounds; the grounds bloom and bubble. [00:03-00:06] Close-up of the lid being placed on top; the silver plunger is held steady at the top for a moment. [00:06-00:08] Slow, steady downward press of the plunger, separating the dark liquid from the grounds. SFX: The gentle gurgle of pouring water and the muffled scrape of the filter.",
  "4.mp4": "[00:00-00:02] Macro shot of five polished steel spheres hanging perfectly still; a finger enters the frame and pulls the far-right sphere back, creating tension. [00:02-00:04] Side view in slow motion as the sphere is released, swinging down and striking the stationary line with a sharp, metallic \"click\". [00:04-00:06] Tight tracking shot following the kinetic energy as the far-left sphere instantly reacts, swinging outward while the middle three remain perfectly still. [00:06-00:08] High-angle wide shot showing the rhythmic, hypnotic back-and-forth motion of the cradle on the desk as the energy begins to dissipate.",
  "5.mp4": "[00:00-00:03] Hands-only close-up; the rope forms a loop (the \"hole\"). [00:03-00:06] The \"rabbit\" (rope end) comes up through the hole, goes around the \"tree,\" and back down the hole. [00:06-00:08] Hands pull both ends to tighten, showing the perfect, secure knot structure. SFX: The rhythmic friction sound of heavy rope sliding against itself.",
  "7.mp4": "[00:00-00:02] Street-level shot focusing on the tire of a bright yellow taxi as it rolls over a manhole cover emitting steam. [00:02-00:05] Tracking shot following the taxi's side door, reflecting the neon billboards and crowds of Times Square in the polished yellow paint. [00:05-00:08] Wide, low-angle shot as the taxi merges into traffic, with the massive digital screens of the city towering above. SFX: A chaotic symphony of car horns, muffled chatter, and the hum of a city that never sleeps.",
  "8.mp4": "[00:00-00:03] Eye-level shot of a person’s feet stepping onto the white stripes of the iconic zebra crossing. [00:03-00:06] Side-profile tracking shot of the person walking across with a relaxed gait, mimicking the famous album cover pose. [00:06-00:08] Wide shot from the center of the road, showing the person reaching the other side as a black cab waits patiently in the background. SFX: The distant chirp of a London \"pelican\" crossing signal.",
  "9.mp4": "[00:00-00:03] Wide landscape shot of the vast steppe; a rider gallops into frame holding a long wooden pole (uurga). [00:03-00:06] Tracking shot alongside the horseman as he maneuvers the leather loop at the end of the pole over a wild horse’s neck. [00:06-00:08] The rider leans back, tension tightening the loop as the wild horse slows to a halt. SFX: Thundering hooves on grass and the whistling of the wind.",
  "10.mp4": "[00:00-00:03] Low-angle shot of the samurai’s feet firmly planted in the dirt; the long scabbard of the nodachi rests against their hip. [00:03-00:06] Medium shot as the samurai unsheathes the massive blade in one fluid motion, the steel gleaming under a dim sun. [00:06-00:08] Wide shot of the samurai holding a high guard (Jodan-no-kamae), the wind whipping through their traditional robes. Emotion: Focused discipline and lethal calm.",
  "11.mp4": "[00:00-00:03] Close-up of a bright green chrysalis hanging from a leaf, dotted with tiny gold spots. [00:03-00:06] Time-lapse effect: the casing becomes transparent, revealing the orange and black wing patterns inside. [00:06-00:08] The casing begins to split at the bottom as the butterfly starts its first movement. SFX: Soft, ambient forest sounds.",
  "12.mp4": "[00:00-00:03] Close-up of a chef’s floured hands slamming a thick rope of dough onto a wooden table. [00:03-00:06] Wide shot as the chef stretches the dough wide, then loops and pulls it again, the strands doubling in number instantly. [00:06-00:08] Close-up of the finished, paper-thin noodles being dropped into a pot of boiling, steaming broth. SFX: The rhythmic \"thwack\" of dough hitting the counter.",
  "13.mp4": "[00:00-00:03] Extreme close-up of a hand with steady fingers lightly tapping a middle block in the wooden tower. [00:03-00:06] Close-up as the block slides out millimeter by millimeter; the tower above it wobbles slightly but holds. [00:06-00:08] Wide shot of the person successfully placing the block on the very top of the trembling tower. SFX: The dry, hollow \"clack\" of wood on wood.",
  "14.mp4": "[00:00-00:03] Close-up of a hand slowly turning the stopcock of a glass burette; a single drop falls into a clear solution. [00:03-00:06] Medium shot of the flask being swirled; the clear liquid flashes pink for a second then disappears. [00:06-00:08] Extreme close-up of the \"End Point\": one final drop turns the entire solution a persistent, pale lilac pink. SFX: The rhythmic \"drip... drip... drip\" into the glass flask.",
  "15.mp4": "[00:00-00:03] First-person view of a hand passing a box of cereal over the red laser of the scanner. [00:03-00:06] Close-up of the digital screen updating the price, followed by a hand tapping a \"Pay Now\" button. [00:06-00:08] Medium shot of the person bagging their items as the receipt begins to print. SFX: The electronic \"beep\" of a successful scan and the mechanical whir of the printer.",
  "16.mp4": "[00:00-00:02] Close-up of a finger flipping the \"On\" switch; the heavy platter begins to rotate slowly. [00:02-00:05] Side view of the tonearm being lifted and carefully moved over the outer edge of the spinning black vinyl. [00:05-00:08] Extreme close-up of the needle dropping into the groove, followed by the slight vibration of the arm. SFX: The warm, crackling static of the needle finding the groove, followed by the first note of music.",
  "17.mp4": "[00:00-00:03] Wide shot of a person placing a jack under a car on the roadside; they begin to pump the handle. [00:03-00:06] Close-up of the lug nuts being loosened with a wrench and removed by hand. [00:06-00:08] The flat tire is pulled off the hub, revealing the brake rotor behind it. SFX: Clinking metal tools and the rhythmic squeak of the jack.",
  "18.mp4": "[00:00-00:03] Low-angle medium shot of a dancer slapping their thighs and chest with intense energy. [00:03-00:06] Extreme close-up of the \"Pukana\" expression: eyes wide, tongue protruding in a fierce grimace. [00:06-00:08] Wide shot of the final unified stomp, hands reaching toward the sky. Emotion: Fierce pride and ancestral power.",
  "19.mp4": "[00:00-00:03] Low-angle shot of a woman’s feet in pointed silk shoes walking across a cold stone floor. [00:03-00:06] Tracking shot following the tall, conical \"Hennin\" headdress, with its long white veil trailing elegantly behind her. [00:06-00:08] Wide shot as she enters a candlelit court, her silhouette elongated by the towering headpiece against the arched windows. SFX: The rustle of heavy velvet and the echo of footsteps on stone.",
  "20.mp4": "[00:00-00:03] Macro shot of a colorful Mantis Shrimp inside a glass tank, its \"raptorial claws\" folded tightly. [00:03-00:05] Super slow-motion shot of the strike; the claw snaps forward with a visible cavitation bubble in the water. [00:05-00:08] The glass surface shows a spiderweb crack from the point of impact. SFX: A sharp, underwater \"crack\" like a gunshot.",
  "21.mp4": "[00:00-00:03] Close-up of a knife spreading thick, creamy peanut butter onto a slice of white bread. [00:03-00:06] Close-up of a second knife spreading bright purple grape jelly onto the opposite slice. [00:06-00:08] Slow-motion shot of the two slices being pressed together, the fillings meeting at the edges. SFX: The sticky, squelching sound of the spread and the crusty scrape of the knife.",
  "22.mp4": "[00:00-00:03] Medium shot of a person standing before a blue USPS mailbox on a sunny sidewalk. [00:03-00:06] Close-up of the hand pulling the heavy metal handle down, revealing the dark interior. [00:06-00:08] The letter is dropped in, and the handle is released, snapping shut with a distinct metallic sound. SFX: The heavy \"clunk-shush\" of the mailbox door closing.",
  "23.mp4": "[00:00-00:03] Close-up of a carbon fiber strip resting on two metal supports in a lab setting. [00:03-00:06] A central hydraulic press descends slowly, bowing the material into a deep \"U\" shape. [00:06-00:08] The pressure increases until the fibers splinter, captured in high-detail macro. SFX: The low hum of the hydraulic press followed by a sharp, composite \"snap.\"",
  "24.mp4": "[00:00-00:03] Close-up of a copper cezve being pushed deep into a bed of glowing, fine hot sand. [00:03-00:06] Medium shot as the dark coffee inside the cezve begins to froth and rise rapidly to the brim. [00:06-00:08] Close-up of the thick, dark liquid being poured into a small ornate cup, topped with a rich layer of foam. SFX: A soft, continuous sizzle of the sand against the copper.",
  "25.mp4": "[00:00-00:03] Side view of an archer at full draw, the bow strained and the arrow tip perfectly still. [00:03-00:06] Slow-motion shot of the release; as the arrow flies, the archer’s bow-hand snaps forward and down (the Khatra). [00:06-00:08] Close-up of the bow spinning slightly in the archer’s relaxed grip as the arrow clears the riser. SFX: The sharp \"twang\" of the bowstring and the whistle of the arrow through air.",
  "26.mp4": "[00:00-00:03] Tabletop view of a broken ceramic bowl; a fine brush applies gold-dusted lacquer to a jagged edge. [00:03-00:06] Close-up as two pieces are carefully joined, the gold seam glowing under a soft lamp. [00:06-00:08] The finished bowl is rotated slowly, showing the beautiful \"scars\" that make it stronger. SFX: Delicate brush strokes and the soft clink of ceramic."
};

const set2: PromptMap = {
  "0.mp4": "[00:00-00:02] Top-down view of a sealed, minimalist Apple box on a dark wooden table. Two hands enter the frame and slowly slide the lid upward, showing a tight friction fit.\n[00:02-00:04] Tight close-up as the lid is removed to reveal the iPhone 16 Pro face-down. The Desert Titanium finish catches the soft studio light, emphasizing its metallic texture.\n[00:04-00:06] Side-angle shot as a hand lifts the phone, highlighting the thin profile and the triple-lens camera bump. Underneath, a coiled USB-C cable and documentation are visible.\n[00:06-00:08] Macro shot of a thumb peeling back the protective film from the screen in one slow, satisfying motion.\nSFX: The smooth slide of cardboard, subtle paper rustle, and a crisp, high-quality plastic peel ASMR.",
  "1.mp4": "[00:00-00:03] Low-angle ground shot. The sharp, geometric stainless-steel body of a Cybertruck approaches the camera, reflecting the environment on its flat panels.\n[00:03-00:06] Slow-motion close-up on the front tire as it strikes a large puddle. Water explodes outward in a dramatic arc, showing massive displacement.\n[00:06-00:08] Tracking shot following the truck as it accelerates smoothly through the resistance, water trailing from the tires.\nSFX: A deep electric motor hum followed by a heavy, cinematic \"whoosh\" of water.",
  "2.mp4": "[00:00-00:03] Side-view medium shot of a person holding a Coca-Cola bottle. Fingers grip the cap and twist it counterclockwise.\n[00:03-00:05] Close-up on the cap. It separates from the ring but remains attached via a plastic hinge, bending back to rest against the side of the bottle.\n[00:05-00:08] The person tilts the bottle to take a drink, showing how the hinged mechanism prevents the cap from detaching or getting lost.",
  "3.mp4": "[00:00-00:02] Close-up of coarse coffee grounds being poured into a clear glass carafe. Steam rises as hot water is poured over them, creating a \"bloom\" of bubbles.\n[00:02-00:04] A wooden spoon stirs the mixture gently before the lid with the plunger is placed on top.\n[00:04-00:07] Time-lapse style shot showing the 4-minute wait, followed by a slow, steady hand pressing the plunger down to the bottom.\n[00:07-00:08] Side view of the dark, rich coffee being poured into a ceramic cup.",
  "4.mp4": "[00:00-00:03] Macro shot of five static metal balls aligned on a desk. A hand lifts the outermost ball and releases it.\n[00:03-00:05] Slow-motion impact. The energy transfers through the middle balls, and the opposite ball swings outward in a perfect arc.\n[00:05-00:08] Wide shot showing the repeating, rhythmic oscillation and the visual symmetry of the elastic collisions.",
  "5.mp4": "[00:00-00:03] Hands-only close-up. A small loop (the \"rabbit hole\") is formed in a thick rope. The end of the rope passes up through the loop.\n[00:03-00:06] The end is wrapped around the standing part of the rope (the \"tree\") and fed back down through the original loop.\n[00:06-00:08] The hands pull both ends tight, forming a secure, non-slip loop.",
  "7.mp4": "[00:00-00:04] Wide shot of a neon-lit Times Square at night. A bright yellow taxi enters the frame, with LED billboards reflecting off its hood.\n[00:04-00:08] Street-level tracking shot as the taxi passes the camera, contrasting the iconic yellow against the high-density urban motion and blue/pink neon lights.",
  "8.mp4": "[00:00-00:04] Eye-level static shot of the famous Abbey Road zebra crossing. Strong perspective lines lead toward the background where cars are stopped.\n[00:04-00:08] A person steps onto the white stripes and walks from the sidewalk to the center of the road, mimicking the iconic Beatles album cover.",
  "9.mp4": "[00:00-00:04] Wide landscape shot of the Mongolian steppe. A rider on horseback gallops at high speed, holding a long pole called an \"uurga\".\n[00:04-00:08] The rider extends the uurga, lowering the loop toward a wild horse to capture it, demonstrating incredible balance and coordination.",
  "10.mp4": "[00:00-00:04] A samurai stands in a wide, grounded stance in a traditional dojo. He draws a massive, long-bladed nodachi.\n[00:04-00:08] With a controlled motion, he performs a powerful overhead swing, finishing in a disciplined follow-through and reset.",
  "11.mp4": "[00:00-00:04] Close-up of a hanging green Monarch chrysalis. Subtle movements occur inside as the outer shell begins to turn transparent.\n[00:04-00:08] The shell clears completely, revealing the orange and black patterns of the butterfly's wings folded inside, just before the final transformation.",
  "12.mp4": "[00:00-00:04] A chef rolls dough into a cylinder, then begins a repetitive motion of stretching and folding it.\n[00:04-00:08] The strands multiply exponentially with every pull until hundreds of thin noodles are formed and dropped into a pot of boiling water.",
  "13.mp4": "[00:00-00:04] Close-up of a tall Jenga tower. A player’s fingers gently wiggle a middle block to test its stability.\n[00:04-00:08] The block slowly slides out as the tower shifts and wobbles, before the player successfully places the block on top of the stack.",
  "14.mp4": "[00:00-00:04] A chemist slowly turns the stopcock of a burette, allowing single drops of solution to fall into a swirling flask below.\n[00:04-00:08] Tight shot on the liquid in the flask. With one final drop, the clear liquid suddenly flashes into a permanent light pink, signaling the \"end point\".",
  "15.mp4": "[00:00-00:04] A person scans a barcode on a grocery item, resulting in a \"beep.\" They place the item into the bagging area.\n[00:04-00:08] The screen updates with the price; the person taps their card on the terminal, and a receipt begins to print.",
  "16.mp4": "[00:00-00:04] Macro shot of a record on the platter. The turntable begins to spin, and a hand carefully lowers the tonearm.\n[00:04-00:08] The needle makes contact with the groove. A faint crackle is heard right before the music begins to play.",
  "17.mp4": "[00:00-00:03] A person uses a lug wrench to loosen the nuts on a flat tire. A jack then lifts the side of the car off the ground.\n[00:03-00:06] The flat tire is removed and replaced with a spare. The nuts are hand-tightened in a star pattern.\n[00:06-00:08] The car is lowered, and the wrench is used for the final tightening.",
  "18.mp4": "[00:00-00:04] A group of dancers in a powerful stance begin rhythmic stomping and chest slapping in perfect unison.\n[00:04-00:08] Close-up on a dancer's face performing \"pukana\"—wide eyes and tongue gestures—expressing intense power and unity.",
  "19.mp4": "[00:00-00:04] A woman in a 15th-century medieval court walks slowly. She wears a tall, cone-shaped \"hennin\" headdress with a long veil.\n[00:04-00:08] Tracking shot from the side, emphasizing the vertical exaggeration of her silhouette and the fabric flowing behind her.",
  "20.mp4": "[00:00-00:04] Close-up in an aquarium. A mantis shrimp locks its appendage, preparing to strike a piece of glass.\n[00:04-00:08] In an ultra-fast blur, the strike occurs, creating cavitation bubbles that collapse with visible force against the glass.",
  "21.mp4": "[00:00-00:04] Slices of bread are laid out. Peanut butter is spread smoothly on one, and purple jelly is spread on the other.\n[00:04-00:08] The slices are combined and a knife cuts the sandwich diagonally into two perfect triangles.",
  "22.mp4": "[00:00-00:04] A hand holds a white envelope. The person walks up to a classic blue USPS mailbox on a sidewalk.\n[00:04-00:08] The metal slot is pulled open, the letter is dropped inside, and the slot clangs shut.",
  "23.mp4": "[00:00-00:04] A carbon fiber sample is placed on two metal supports in a laboratory machine. A vertical force starts to press down on the center.\n[00:04-00:08] The material bends gradually under the stress until it reaches the fracture point, snapping to measure its ultimate strength.",
  "24.mp4": "[00:00-00:04] A small metal \"cezve\" containing coffee and water is moved in circles through a bed of extremely hot sand.\n[00:04-00:08] As the heat transfers, a thick foam rises slowly to the brim. The barista removes it just before it boils over and pours it into a small cup.",
  "25.mp4": "[00:00-00:04] Slow-motion side view of an archer drawing a traditional bow. Upon release, the bow hand snaps forward and downward.\n[00:04-00:08] The \"khatra\" movement causes the bow to rotate outward, allowing the arrow to clear the riser without interference as it flies toward the target.",
  "26.mp4": "[00:00-00:04] Soft-lit tabletop shot. An artisan applies a thin line of lacquer to the edge of a broken ceramic shard.\n[00:04-00:08] The pieces are joined, and gold powder is carefully dusted over the seam, highlighting the cracks in gold to embrace the beauty of the repair."
};

const set3: PromptMap = {
  "0.mp4": "[00:00-00:02] Top-down shot of sealed Apple box on wooden table. Hands slide lid upward slowly. [00:02-00:04] Close-up reveal of phone face-down, Desert Titanium catching soft light. [00:04-00:06] Hand lifts phone, thin profile and camera bump emphasized. [00:06-00:08] Insert shot: USB-C cable + documentation neatly arranged. Protective film peel (ASMR). Emotion: premium, minimalism.",
  "1.mp4": "[00:00-00:02] Low-angle shot as Cybertruck approaches reflective road. [00:02-00:04] Tire hits puddle → explosive slow-motion water splash. [00:04-00:06] Tracking side shot, water arcs around wheels. [00:06-00:08] Truck exits frame, water trailing. Tone: power, weight, precision.",
  "2.mp4": "[00:00-00:02] Bottle held sideways, label visible. [00:02-00:04] Fingers twist cap → hinge reveals tether. [00:04-00:06] Cap bends back, stays attached. [00:06-00:08] Pour action. Focus: sustainability design.",
  "3.mp4": "[00:00-00:02] Grounds added → hot water poured (bloom bubbles). [00:02-00:04] Stir gently, lid placed. [00:04-00:06] Slow plunger press. [00:06-00:08] Coffee poured into cup. Mood: calm, methodical.",
  "4.mp4": "[00:00-00:02] Static aligned balls. One lifted. [00:02-00:04] Release → impact transfer. [00:04-00:06] Opposite ball swings outward. [00:06-00:08] Repeating rhythm loop. Theme: physics symmetry.",
  "5.mp4": "[00:00-00:02] Form loop (“rabbit hole”). [00:02-00:04] Rope end passes through loop. [00:04-00:06] Wrap around standing part. [00:06-00:08] Tighten → fixed loop formed. Focus: clarity, precision.",
  "7.mp4": "[00:00-00:02] Neon-lit Times Square wide shot. [00:02-00:04] Yellow taxi enters frame. [00:04-00:06] Reflections ripple across car. [00:06-00:08] Taxi exits with crowd movement. Energy: urban chaos.",
  "8.mp4": "[00:00-00:02] Static zebra crossing. [00:02-00:04] Person steps onto stripes. [00:04-00:06] Mid-crossing symmetry shot. [00:06-00:08] Exit frame. Tone: iconic, nostalgic.",
  "9.mp4": "[00:00-00:02] Wide степpe landscape, rider galloping. [00:02-00:04] Pole extends forward. [00:02-00:06] Loop drops toward horse. [00:06-00:08] Capture moment. Emotion: skill, tradition.",
  "10.mp4": "[00:00-00:02] Grounded stance, sword drawn. [00:02-00:04] Controlled overhead swing. [00:04-00:06] Follow-through motion. [00:06-00:08] Reset stance. Tone: discipline.",
  "11.mp4": "[00:00-00:02] Hanging chrysalis close-up. [00:02-00:04] Subtle internal movement. [00:04-00:06] Shell becomes translucent. [00:06-00:08] Butterfly silhouette visible. Theme: transformation.",
  "12.mp4": "[00:00-00:02] Dough stretched + folded. [00:02-00:04] Strands multiply rapidly. [00:04-00:06] Thin noodles formed. [00:06-00:08] Drop into boiling water. Rhythm: skill mastery.",
  "13.mp4": "[00:00-00:02] Tower close-up. [00:02-00:04] Block tested, slight wiggle. [00:04-00:06] Middle block slowly removed. [00:06-00:08] Tower shifts. Tension rises.",
  "14.mp4": "[00:00-00:02] Burette drip into flask. [00:02-00:04] Swirling motion. [00:04-00:06] Drops slow down. [00:06-00:08] Sudden color change. Moment: precision.",
  "15.mp4": "[00:00-00:02] Barcode scan → beep. [00:02-00:04] Item placed in bagging area. [00:04-00:06] Payment tap. [00:06-00:08] Receipt prints. Tone: automation.",
  "16.mp4": "[00:00-00:02] Record placed on platter. [00:02-00:04] Spin begins. [00:04-00:06] Needle lowers. [00:06-00:08] Music starts. Mood: analog warmth.",
  "17.mp4": "[00:00-00:02] Lug nuts loosened. [00:02-00:04] Car lifted. [00:04-00:06] Tire replaced. [00:06-00:08] Tighten in star pattern. Tone: practical skill.",
  "18.mp4": "[00:00-00:02] Group stance. [00:02-00:04] Stomping + chest strikes. [00:04-00:06] Pukana expressions. [00:06-00:08] Unified final pose. Emotion: intensity.",
  "19.mp4": "[00:00-00:02] Medieval court setting. [00:02-00:04] Tall headdress enters frame. [00:04-00:06] Fabric flows. [00:06-00:08] Slow regal walk. Tone: elegance.",
  "20.mp4": "[00:00-00:02] Shrimp locks appendage. [00:02-00:04] Ultra-fast strike. [00:04-00:06] Cavitation bubbles form. [00:06-00:08] Impact on glass. Energy: explosive.",
  "21.mp4": "[00:00-00:02] Bread laid out. [00:02-00:04] Peanut butter spread. [00:04-00:06] Jelly spread. [00:06-00:08] Sandwich assembled + cut.",
  "22.mp4": "[00:00-00:02] Letter in hand. [00:02-00:04] Blue mailbox. [00:04-00:06] Slot opens. [00:06-00:08] Letter drops. Tone: simple action.",
  "23.mp4": "[00:00-00:02] Sample placed on supports. [00:02-00:04] Force applied downward. [00:04-00:06] Material bends. [00:06-00:08] Fracture. Theme: strength limits.",
  "24.mp4": "[00:00-00:02] Coffee + water + sugar mixed. [00:02-00:04] Heated in sand. [00:04-00:06] Foam rises slowly. [00:06-00:08] Pour into cup. Mood: ritual.",
  "25.mp4": "[00:00-00:02] Archer draws bow. [00:02-00:04] Release → bow rotates. [00:04-00:06] Arrow flies. [00:06-00:08] Follow-through emphasized.",
  "26.mp4": "[00:00-00:02] Broken ceramic pieces arranged. [00:02-00:04] Lacquer applied. [00:04-00:06] Gold powder added. [00:06-00:08] Reassembled bowl revealed. Theme: beauty in repair."
};

const labels: Label[] = ["Set 1", "Set 2", "Set 3"];
const setMap: Record<Label, PromptMap> = { "Set 1": set1, "Set 2": set2, "Set 3": set3 };
const files = Object.keys(prompts).sort((a, b) => Number(a.split(".")[0]) - Number(b.split(".")[0]));

function shuffleLabels(items: Label[]): Label[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

type StatCardProps = {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
};

function StatCard({ title, value, subtitle, icon: Icon }: StatCardProps) {
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-muted-foreground">{title}</div>
            <div className="mt-1 text-2xl font-semibold">{value}</div>
            {subtitle ? <div className="mt-1 text-xs text-muted-foreground">{subtitle}</div> : null}
          </div>
          <div className="rounded-2xl border p-3">
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Page() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [choices, setChoices] = useState<Record<string, Label>>({});
  const setOrderByFile = useMemo<Record<string, Label[]>>(() => {
    return files.reduce<Record<string, Label[]>>((acc, file) => {
      acc[file] = shuffleLabels(labels);
      return acc;
    }, {});
  }, []);

  const currentFile = files[currentIndex];
  const answeredCount = Object.keys(choices).length;
  const completion = Math.round((answeredCount / files.length) * 100);

  const counts = useMemo<Record<Label, number>>(() => {
    return labels.reduce<Record<Label, number>>(
      (acc, label) => {
        acc[label] = Object.values(choices).filter((v) => v === label).length;
        return acc;
      },
      { "Set 1": 0, "Set 2": 0, "Set 3": 0 },
    );
  }, [choices]);

  const bestSet = useMemo(() => {
    const ordered = [...labels].sort((a, b) => counts[b] - counts[a]);
    if (counts[ordered[0]] === 0) return "—";
    if (ordered.length > 1 && counts[ordered[0]] === counts[ordered[1]]) return "Tie";
    return ordered[0];
  }, [counts]);

  const choose = (label: Label) => {
    setChoices((prev) => ({ ...prev, [currentFile]: label }));
  };

  const move = (dir: number) => {
    setCurrentIndex((prev) => Math.min(files.length - 1, Math.max(0, prev + dir)));
  };

  const jumpToFirstUnanswered = () => {
    const next = files.findIndex((file) => !choices[file]);
    if (next >= 0) setCurrentIndex(next);
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="grid gap-4 md:grid-cols-4">
          <StatCard title="Prompts" value={files.length} subtitle="Evaluation items" icon={ClipboardList} />
          <StatCard title="Completed" value={`${answeredCount}/${files.length}`} subtitle={`${completion}% finished`} icon={CheckCircle2} />
          <StatCard title="Current leader" value={bestSet} subtitle="Based on selections so far" icon={Trophy} />
          <StatCard title="Set 2 coverage" value={counts["Set 2"]} subtitle="Wins recorded" icon={BarChart3} />
        </div>

        <div className="grid gap-6 lg:grid-cols-[280px_1fr_320px]">
          <Card className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">Prompt list</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="mb-4 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Progress</span>
                  <span>{completion}%</span>
                </div>
                <Progress value={completion} className="h-2" />
                <Button variant="outline" className="w-full rounded-xl" onClick={jumpToFirstUnanswered}>
                  Jump to next unanswered
                </Button>
              </div>
              <ScrollArea className="h-[560px] pr-3">
                <div className="space-y-2">
                  {files.map((file, idx) => {
                    const selected = choices[file];
                    const active = idx === currentIndex;
                    return (
                      <button
                        key={file}
                        onClick={() => setCurrentIndex(idx)}
                        className={`w-full rounded-2xl border p-3 text-left transition ${
                          active ? "border-slate-900 bg-white shadow-sm" : "border-slate-200 bg-white/70 hover:bg-white"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-medium">{file}</div>
                          {selected ? <Badge variant="secondary">{selected}</Badge> : <Badge variant="outline">Open</Badge>}
                        </div>
                        <div className="mt-2 line-clamp-3 text-xs text-muted-foreground">{prompts[file]}</div>
                      </button>
                    );
                  })}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="rounded-2xl shadow-sm">
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="text-sm text-muted-foreground">Now reviewing</div>
                    <CardTitle className="text-2xl">{currentFile}</CardTitle>
                  </div>
                  <Badge className="rounded-xl px-3 py-1 text-sm" variant="secondary">
                    {currentIndex + 1} / {files.length}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="rounded-2xl border bg-white p-4">
                  <div className="mb-2 text-sm font-medium text-muted-foreground">Original prompt</div>
                  <p className="text-base leading-7">{prompts[currentFile]}</p>
                </div>
                <div className="mt-4 flex items-center gap-3">
                  <Button variant="outline" className="rounded-xl" onClick={() => move(-1)} disabled={currentIndex === 0}>
                    <ChevronLeft className="mr-2 h-4 w-4" /> Prev
                  </Button>
                  <Button variant="outline" className="rounded-xl" onClick={() => move(1)} disabled={currentIndex === files.length - 1}>
                    Next <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                  {choices[currentFile] ? <Badge className="rounded-xl">Selected: {choices[currentFile]}</Badge> : null}
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-4 xl:grid-cols-3">
              {setOrderByFile[currentFile].map((label) => {
                const script = setMap[label][currentFile] || "No script provided for this file in this set.";
                const chosen = choices[currentFile] === label;
                return (
                  <Card key={label} className={`rounded-2xl shadow-sm ${chosen ? "ring-2 ring-slate-900" : ""}`}>
                    <CardHeader>
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-lg text-white">{label}</CardTitle>
                        {chosen ? <Badge>Chosen</Badge> : <Badge variant="outline">{counts[label]} wins</Badge>}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <ScrollArea className="h-[330px] rounded-xl border bg-slate-50 p-4">
                        <p className="whitespace-pre-wrap text-sm leading-6">{script}</p>
                      </ScrollArea>
                      <Button className="w-full rounded-xl" onClick={() => choose(label)}>
                        Choose {label}
                      </Button>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          <Card className="rounded-2xl shadow-sm">
            <CardHeader>
              <CardTitle className="text-base">Evaluator stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-0">
              {labels.map((label) => {
                const pct = answeredCount ? Math.round((counts[label] / answeredCount) * 100) : 0;
                return (
                  <div key={label} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>{label}</span>
                      <span className="font-medium">
                        {counts[label]} wins · {pct}%
                      </span>
                    </div>
                    <Progress value={pct} className="h-2" />
                  </div>
                );
              })}

              <Separator />

              <div>
                <div className="mb-3 text-sm font-medium">Selections by prompt</div>
                <ScrollArea className="h-[360px] pr-3">
                  <div className="space-y-2">
                    {files.map((file) => (
                      <div key={file} className="rounded-2xl border bg-white p-3">
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-medium">{file}</div>
                          {choices[file] ? <Badge>{choices[file]}</Badge> : <Badge variant="outline">Pending</Badge>}
                        </div>
                        <div className="mt-2 line-clamp-2 text-xs text-muted-foreground">{prompts[file]}</div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
