import React, { Fragment } from 'react'
import { Highlight, Snippet } from 'react-instantsearch-dom'
import { Calendar } from 'styled-icons/octicons/Calendar'
import { Tags } from 'styled-icons/fa-solid/Tags'

// export const PageHit = clickHandler => ({ hit }) => {
//   console.log(hit)
//   return (
//     <div>
//       <Link to={hit.slug} onClick={clickHandler}>
//         <h4>
//           <Highlight attribute="title" hit={hit} tagName="mark" />
//         </h4>
//       </Link>
//       <Snippet attribute="excerpt" hit={hit} tagName="mark" />
//     </div>
//   )
// }

export const PostHit = clickHandler => ({ hit }) => {
  console.log(hit)
  return (
    <div>
      <a href={`${hit.fields.slug}`} target="_blank" onClick={clickHandler}>
        <h4>
          <Highlight attribute="title" hit={hit} tagName="mark" />
        </h4>
      </a>
      <div>
        <Calendar size="1em" />
        &nbsp;
        <Highlight attribute="date" hit={hit} tagName="mark" />
        &emsp;
        <Tags size="1em" />
        &nbsp;
        {/* {hit.tags.map((tag, index) => (
        <Fragment key={tag}>
          {index > 0 && `, `}
          {tag}
        </Fragment>
      ))} */}
        <Fragment>{hit.category}</Fragment>
      </div>
      <Snippet attribute="excerpt" hit={hit} tagName="mark" />
    </div>
  )
}
