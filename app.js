console.log("戦闘力スキャナ起動");

function randomPower() {
  let base = Math.random();

  let rarity = 1;
  if (base < 0.005) rarity = 1.3;
  else if (base < 0.05) rarity = 1.15;
  else rarity = 1;

  let power = Math.floor((base * base) * 1000000 * rarity);

  alert("戦闘力は " + power.toLocaleString() + " です！");
}
