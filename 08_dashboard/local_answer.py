
"""Fast explanations for verified RetailIQ AI sales calculations."""


def format_verified_answer(result):
    """Explain approved sales calculations without calling Gemini."""

    topic = result.get("topic")
    evidence = result.get("evidence", [])

    if not evidence:
        return None

    records = evidence[0].get("records", [])

    if not records:
        return None

    # Explain store and department rankings.
    if topic in {
        "custom_store_ranking",
        "custom_department_ranking",
    }:
        entity_field = (
            "Store"
            if topic == "custom_store_ranking"
            else "Dept"
        )

        entity_name = (
            "Store"
            if entity_field == "Store"
            else "Department"
        )

        description = evidence[0].get("description", "").lower()

        direction = (
            "lowest"
            if "lowest" in description
            else "highest"
        )

        lines = [
            f"Here are the {len(records)} {direction}-selling "
            f"{entity_name.lower()} results based on historical "
            "total sales:",
        ]

        for index, record in enumerate(records, start=1):
            entity_id = int(record[entity_field])
            sales = float(record["Total_Sales"])

            lines.append(
                f"{index}. **{entity_name} {entity_id}** — "
                f"total sales: **{sales:,.2f}**"
            )

        lines.append(
            "\nThese rankings come from the saved RetailIQ AI "
            "sales report. See the chart and source records below."
        )

        return "\n\n".join(lines)

    # Explain comparisons between two stores.
    if topic == "custom_store_comparison":
        if len(records) != 2:
            return None

        first, second = records

        first_store = int(first["Store"])
        second_store = int(second["Store"])

        first_sales = float(first["Total_Sales"])
        second_sales = float(second["Total_Sales"])

        difference = float(
            first["Absolute_Sales_Difference"]
        )

        higher = first["Higher_Sales_Store"]

        if higher is None:
            conclusion = "Both stores recorded equal total sales."
        else:
            conclusion = (
                f"**Store {int(higher)}** recorded higher "
                "historical total sales."
            )

        return (
            f"### Store {first_store} vs Store {second_store}\n\n"
            f"- Store {first_store}: **{first_sales:,.2f}**\n"
            f"- Store {second_store}: **{second_sales:,.2f}**\n"
            f"- Absolute difference: **{difference:,.2f}**\n\n"
            f"{conclusion}\n\n"
            "The comparison uses the verified historical "
            "store performance report."
        )

    return None
